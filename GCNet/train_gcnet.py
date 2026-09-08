def main():
    # load necessary modules
    from engine.engine_base import Engine
    from config.config import GCNetconfig

    # set up DNN
    config = GCNetconfig()
    engine = Engine(config)

    # prepare dataset
    data_loader_train, data_loader_validation, data_loader_test = (
        engine.prepare_dataset()
    )

    # train
    engine.train_v2(data_loader_train, data_loader_validation)

    # plot learning curve
    save_flag = 1
    engine.plotLine_learning_curve_v2(save_flag)

    # plot learning rate
    engine.plot_learning_rate(data_loader_train, save_flag)


if __name__ == "__main__":
    main()
