"""Training entry point; safe for spawned DataLoader workers."""


def main():
    from engine.engine_bnn import EngineBNN
    from config.config_bnn import ConfigBNN

    # set up DNN
    config = ConfigBNN()
    engine = EngineBNN(config)

    # prepare dataset
    train_loader, validation_loader, test_loader = engine.prepare_dataset()

    # train
    engine.train(train_loader, validation_loader, test_loader)

    # plot learning curve
    save_flag = 1
    engine.plotLine_learning_curve(save_flag)

    # plot learning rate
    engine.plot_learning_rate(train_loader, save_flag)
    # engine.inference_val(validation_loader)


if __name__ == "__main__":
    main()
