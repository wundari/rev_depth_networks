import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import warnings
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Sequence, Tuple, Union, Optional
from collections import defaultdict
from jaxtyping import Float
from torch import Tensor

from utilities.misc import NestedTensor
from engine.engine_bnn import EngineBNN
from engine.engine_gcnet_lr import EngineGCNet
from config.config_am import ConfigAM

# Enable TF32 for RTX 5090 optimizations (Ablation/Inference is generally stable with TF32)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class ChannelAblationHook:
    """
    Forward hook to ablate (zero-out or mean-clamp) specific weakly-coupled channels 
    in a layer's output feature map during inference.
    """
    def __init__(self, channels_to_ablate: List[int], clamp_values: Optional[Dict[int, float]] = None):
        self.channels_to_ablate = channels_to_ablate
        self.clamp_values = clamp_values

    def __call__(self, module: nn.Module, inputs: Tuple[Tensor], output: Tensor) -> Tensor:
        if not self.channels_to_ablate:
            return output
            
        # Clone to avoid in-place modification errors in autograd/ReLU graphs
        ablated_output = output.clone()
        
        for ch in self.channels_to_ablate:
            if self.clamp_values and ch in self.clamp_values:
                ablated_output[:, ch] = self.clamp_values[ch]
            else:
                ablated_output[:, ch] = 0.0
                
        return ablated_output


class MonocularTracer:
    """
    Identifies strongly vs weakly coupled neurons via monocular/binocular stimulation 
    and evaluates network performance when weakly-coupled neurons are ablated.
    """
    def __init__(self, engine: Union[EngineBNN, EngineGCNet], config: ConfigAM):
        self.engine = engine
        self.model = engine.model
        self.config = config
        self.device = next(self.model.parameters()).device
        self.baseline_val = config.baseline if hasattr(config, 'baseline') else 0.5
        
    def _get_blank_input(self, template: Tensor) -> Tensor:
        return torch.full_like(template, self.baseline_val)

    @torch.inference_mode()
    def compute_coupling_indices(
        self, 
        dataloader, 
        target_layers: Dict[str, str], 
        num_batches: int = 10
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Profiles layers dynamically over a dataset to classify channels based on ocular dominance.
        Records mean activations for: Both, Left-only, Right-only, Blank.
        """
        self.model.eval()
        
        # Store accumulated spatial means per channel
        # Format: stats[layer_name][condition] = Tensor of shape [C]
        stats = defaultdict(lambda: defaultdict(float))
        counts = 0

        # Hook to record channel-wise means
        active_condition = "both"
        
        def get_record_hook(layer_name):
            def hook(module, inp, out):
                # Spatial mean across H, W (and D if 5D cost volume)
                dims_to_reduce = list(range(2, out.ndim))
                spatial_mean = out.mean(dim=dims_to_reduce).sum(dim=0) # Sum over batch
                
                if stats[layer_name][active_condition] is float:
                    stats[layer_name][active_condition] = spatial_mean.cpu()
                else:
                    stats[layer_name][active_condition] += spatial_mean.cpu()
            return hook

        handles = []
        for name, mod_name in target_layers.items():
            mod = self.model.get_submodule(mod_name)
            handles.append(mod.register_forward_hook(get_record_hook(name)))

        print(f"Profiling coupling on {len(target_layers)} layers over {num_batches} batches...")
        
        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            left_img = data['left'].to(self.device)
            right_img = data['right'].to(self.device)
            ref = data.get('ref', torch.ones(left_img.shape[0], device=self.device))
            blank_img = self._get_blank_input(left_img)

            conditions = {
                'both': (left_img, right_img),
                'left_only': (left_img, blank_img),
                'right_only': (blank_img, right_img),
                'blank': (blank_img, blank_img)
            }

            for cond_name, (l_img, r_img) in conditions.items():
                active_condition = cond_name
                self.model(NestedTensor(left=l_img, right=r_img, ref=ref))
                
            counts += left_img.shape[0]

        # Cleanup hooks and average
        for h in handles: 
            h.remove()
            
        final_stats = {}
        for layer, cond_data in stats.items():
            final_stats[layer] = {cond: val / counts for cond, val in cond_data.items()}
            
        return final_stats

    def identify_weakly_coupled(
        self, 
        coupling_stats: Dict[str, Dict[str, Tensor]], 
        interaction_threshold: float = 0.05,
        activity_threshold: float = 0.01
    ) -> Dict[str, List[int]]:
        """
        Identifies weakly coupled channels.
        A channel is weakly coupled if:
        1. Binocular Interaction (Synergy) is near zero (linear summation).
        2. Monocular response (Left or Right) is significantly above blank baseline.
        """
        weak_channels = {}
        
        for layer_name, stats in coupling_stats.items():
            R_both = stats['both']
            R_L = stats['left_only']
            R_R = stats['right_only']
            R_blank = stats['blank']
            
            # Synergy: Interaction = R_both - R_L - R_R + R_blank
            interaction = torch.abs(R_both - R_L - R_R + R_blank)
            
            # Monocular drive strength
            drive_L = R_L - R_blank
            drive_R = R_R - R_blank
            max_monocular_drive = torch.maximum(drive_L, drive_R)
            
            # Weakly coupled: Low interaction synergy BUT high monocular activity
            is_weak = (interaction < interaction_threshold) & (max_monocular_drive > activity_threshold)
            weak_indices = torch.nonzero(is_weak).squeeze(-1).tolist()
            
            # Ensure it's a list even if 1 element
            if isinstance(weak_indices, int):
                weak_indices = [weak_indices]
                
            weak_channels[layer_name] = weak_indices
            print(f"Layer {layer_name}: Found {len(weak_indices)} weakly coupled channels.")
            
        return weak_channels

    @torch.inference_mode()
    def evaluate_with_ablation(
        self, 
        dataloader, 
        target_layers: Dict[str, str], 
        weak_channels: Dict[str, List[int]]
    ) -> Dict[str, float]:
        """
        Ablates weakly coupled channels simultaneously (or layer-by-layer) and evaluates 
        downstream performance (e.g., EPE, >3px error) on the test dataset.
        """
        handles = []
        for name, mod_name in target_layers.items():
            channels_to_ablate = weak_channels.get(name, [])
            if channels_to_ablate:
                mod = self.model.get_submodule(mod_name)
                # Using 0.0 for ablation. Consider dataset mean-clamping to avoid covariate shift.
                hook = ChannelAblationHook(channels_to_ablate)
                handles.append(mod.register_forward_hook(hook))
                
        print(f"Running evaluation with {len(handles)} active ablation hooks...")
        try:
            # Assuming the engine exposes a validate/test method 
            # Replace `self.engine.validate` with your exact evaluation call
            metrics = self.engine.validate(dataloader) 
        finally:
            # Guarantee hooks are removed even if evaluation fails
            for h in handles:
                h.remove()
                
        return metrics

def run_tracer(engine, config_am, test_dataloader):
    tracer = MonocularTracer(engine, config_am)
    
    # Define which layers to trace (Example matching GC_Net_LR)
    target_layers = {
        'decoder_layer21_pre': 'decoder.layer21.0',
        'decoder_layer24_pre': 'decoder.layer24.0',
        'decoder_layer27_pre': 'decoder.layer27.0'
    }
    
    # 1. Profile dataset coupling
    coupling_stats = tracer.compute_coupling_indices(test_dataloader, target_layers, num_batches=20)
    
    # 2. Identify monocularly-driven (weakly coupled) neurons
    weak_channels = tracer.identify_weakly_coupled(coupling_stats)
    
    # 3. Baseline Evaluation (No Ablation)
    print("Evaluating Baseline...")
    baseline_metrics = tracer.evaluate_with_ablation(test_dataloader, target_layers, weak_channels={})
    
    # 4. Ablation Evaluation
    print("Evaluating with Monocular Ablation...")
    ablated_metrics = tracer.evaluate_with_ablation(test_dataloader, target_layers, weak_channels)
    
    print("Baseline Metrics:", baseline_metrics)
    print("Ablated Metrics:", ablated_metrics)

if __name__ == '__main__':
    # Entry point placeholder - expects engine and dataloader to be instantiated
    pass
