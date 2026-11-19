from config.config_bnn import ConfigBNN
from config.config_bi import ConfigBI
from engine.engine_bnn import EngineBNN
from Binocular_integration.bino_integration import BinocularIntegration


def main():

    config = ConfigBNN()
    engine = EngineBNN(config)
    config.load_state = True
    config_bi = ConfigBI()

    bi = BinocularIntegration(engine, config_bi)
    bi.config.batch_size = 8
    bi.config.batch_size_val = 8

    # prepare dataset
    train_loader, validation_loader, test_loader = engine.prepare_dataset()

    # compute integration ratio and count
    bi.compute_bino_integration(train_loader)
    bi.plotLine_integration("ratio")
    bi.plotLine_integration("count")


# %%
if __name__ == "__main__":
    main()
