"""Training entry point; safe for spawned DataLoader workers."""


def main():
    from engine.engine_base import Engine
    from config.config import BNNconfig

    config = BNNconfig()
    engine = Engine(config)
    train_loader, validation_loader, test_loader = engine.prepare_dataset()
    engine.train_v2(train_loader, validation_loader)
    save_flag = 1
    engine.plotLine_learning_curve_v2(save_flag)
    engine.plot_learning_rate(train_loader, save_flag)
    # engine.inference_val(validation_loader)


if __name__ == "__main__":
    main()
