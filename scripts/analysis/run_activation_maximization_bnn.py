def main():
    from FeatureVis.feat_vis import ActivationMaximization, RGBNormalize
    from config.config_am import ConfigAM
    from config.config_bnn import ConfigBNN
    from engine.engine_bnn import EngineBNN

    import torch

    config = ConfigBNN()
    config.load_state = True
    config.compile_mode = None  # prevent unexpected results
    config_am = ConfigAM()
    config_am.size_multiple = 2
    engine = EngineBNN(config)

    am = ActivationMaximization(engine, config_am)

    normalizer = RGBNormalize().to(config.device)
    safe_flag = True

    for layer in am.target_layers:

        module = am.model.get_submodule(layer["module"])
        channels = layer["channels"]
        if channels is None:
            if not hasattr(module, "out_channels"):
                raise ValueError(
                    "Set channels explicitly for composite/nonconvolution modules"
                )
            channels = list(range(module.out_channels))

        # update config_am
        config_am.input_size = layer.get("input_size")
        config_am.disp_idx = layer.get("disp_idx")
        config_am.disp_px = layer.get("disp_px")

        # start activation maximization
        result = am.maximize_layer(
            module,
            channels,
            normalizer,
            call_index=layer["call"],
            layer_name=layer["module"],
            device=torch.device("cuda:0"),
        )
        result["target_spec"] = layer

        # save data
        if safe_flag:
            filename = f"{am.am_dir}/{layer["name"]}.pt"
            torch.save(result, filename)

        # plot
        # per-unit pages: curve | left | right | sensitivity crops | eye ablation
        filename = f"{am.am_dir}/{layer["name"]}.pdf"
        am.plot_layer(layer["name"], result, filename)

        # one population overview of the eye ablation for this layer
        filename = f"{am.am_dir}/{layer["name"]}_ablation_summary.pdf"
        am.plot_ablation_summary(layer["name"], result, filename)


# %%
if __name__ == "__main__":
    main()
