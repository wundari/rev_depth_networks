def main():
    # load necessary modules
    from engine.engine_gcnet import EngineGCNet
    from GC_Net.config.config import ConfigGCNet

    # set up DNN
    config = ConfigGCNet()
    engine = EngineGCNet(config)

    # prepare dataset
    train_loader, validation_loader, test_loader = engine.prepare_dataset()

    # train
    engine.train(train_loader, validation_loader, test_loader)

    # plot learning curve
    save_flag = 1
    engine.plotLine_learning_curve(save_flag)

    # plot learning rate
    engine.plot_learning_rate(train_loader, save_flag)


if __name__ == "__main__":
    main()
