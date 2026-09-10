from typing import Callable, Iterable, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from utilities.typing import ModuleOutputMapping


class ModuleOutputsHook:
    """
    refs:
    https://distill.pub/2020/circuits/visualizing-weights/
    https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
    https://www.kaggle.com/code/sironghuang/understanding-pytorch-hooks/notebook
    https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5?u=ptrblck
    https://discuss.pytorch.org/t/how-to-register-forward-hooks-for-each-module/43347
    """

    def __init__(self, target_modules: Iterable[nn.Module]) -> None:
        # self.hooks = {}
        # for module_name, module in model.named_modules():
        #     self.hooks[module_name] = module.register_forward_hook(
        #         self._forward_hook(module_name)
        #     )
        self.activations: ModuleOutputMapping = dict.fromkeys(target_modules, None)
        self.hooks = [
            module.register_forward_hook(self._forward_hook())
            for module in target_modules
        ]

    def _reset_outputs(self) -> None:
        """
        Delete captured activations.
        """
        self.activations = dict.fromkeys(self.activations.keys(), None)

    # @property
    # def is_ready(self) -> bool:
    #     return all(value is not None for value in self.outputs.values())

    @property
    def targets(self) -> Iterable[nn.Module]:
        return self.activations.keys()

    def _forward_hook(self) -> Callable:
        """
        Return the module_outputs_forward_hook forward hook function.

        Returns:
            forward_hook (Callable): The module_outputs_forward_hook function.
        """

        def module_outputs_forward_hook(
            module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            self.activations[module] = output

        return module_outputs_forward_hook

    def consume_outputs(self) -> ModuleOutputMapping:
        """
        Collect model layers' activations and return them.

        Returns:
            outputs (ModuleOutputMapping): The captured outputs.
        """
        activations = self.activations
        self._reset_outputs()

        return activations

    def remove_hooks(self) -> None:
        """
        Remove hooks.
        """

        for hook in self.hooks:
            hook.remove()

    def __del__(self) -> None:
        """
        Ensure that using "del" properly deletes hooks
        """
        self.remove_hooks()


# class Hook:
#     def __init__(self, module, backward=False):
#         if backward == False:
#             self.hook = module.register_forward_hook(self.hook_fn)
#         else:
#             self.hook = module.register_backward_hook(self.hook_fn)

#     def hook_fn(self, module, input, output):
#         self.input = input  # previous layer's output
#         self.output = output  # current layer's output

#     def close(self):
#         self.hook.remove()


# hookF = [Hook(layer[1]) for layer in list(model._modules.items())]
# hookF = [Hook(module) for module in list(model.layer18)]


# # compute output model
# out = model(input_left, input_right)

# # get module output
# outputs = []
# for h in hookF:
#     outputs.append(h.output)

# activation = {}


# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()

#     return hook
