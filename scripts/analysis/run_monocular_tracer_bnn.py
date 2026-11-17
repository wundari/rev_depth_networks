def main():
    from MonocularTracer.monocular_tracer import MonocularTracer
    from config.config_monocular_tracer import ConfigMonocularTracer
    from config.config_bnn import ConfigBNN
    from engine.engine_bnn import EngineBNN

    # load the model using the same engine as activation maximization
    config = ConfigBNN()
    config.load_state = True
    config.compile_mode = None  # hooks need the original model modules
    config.binocular_interaction = "default"  # default, bem, cmm, or sum_diff
    # config.seed = 1618
    # config.dataset_directory = "/path/to/SceneFlow_complete/Monkaa"
    # Keep architecture and preprocessing settings identical to training.

    # analysis parameters; edit these directly, as in your existing scripts
    config_mt = ConfigMonocularTracer()
    config_mt.unit = "channel"
    config_mt.inference_batch_size = 2  # independent GPU jobs; try 4 if memory permits
    config_mt.num_workers = 4  # load/decode images in parallel on CPU
    config_mt.reuse_encoder_features = True
    config_mt.amp_dtype = "float32"  # optional "bfloat16"; validate before final analysis
    config_mt.fractions = (0.0, 0.1, 0.25, 0.5)
    config_mt.replacement = "zero"
    # config_mt.max_calibration_samples = 32  # optional small pilot
    # config_mt.max_test_samples = 32
    # config_mt.save_figures = False
    # config_mt.target_layers = [
    #     dict(name="layer3", module="decoder.layer3", call=0),
    #     dict(name="layer4", module="decoder.layer4", call=0),
    # ]

    engine = EngineBNN(config)
    tracer = MonocularTracer(engine, config_mt)
    calibration_loader, test_loader = tracer.make_data_loaders()
    safe_flag = True  # follows the save flag in your activation-maximization script

    for layer in tracer.target_layers:

        # first: measure binocular and monocular responses using validation data
        profile_result = tracer.profile_layer(layer, calibration_loader)
        if safe_flag:
            tracer.save_result(layer, profile_result, analysis="profile")

        # second: switch off fixed units and evaluate the stereo test images
        ablation_result = tracer.ablate_layer(layer, profile_result, test_loader)
        if safe_flag:
            tracer.save_result(layer, ablation_result, analysis="ablation")

        # plot this layer's population responses and lesion-performance curves
        if config_mt.save_figures:
            tracer.plot_layer(layer, profile_result)
            tracer.plot_ablation_summary(layer, ablation_result)


# %%
if __name__ == "__main__":
    main()
