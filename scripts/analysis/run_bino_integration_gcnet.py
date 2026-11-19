from config.config_gcnet_lr import ConfigGCNet
from config.config_bi import ConfigBI
from engine.engine_gcnet_lr import EngineGCNet
from Binocular_integration.bino_integration import BinocularIntegration


def main():

    config = ConfigGCNet()
    engine = EngineGCNet(config)
    config.load_state = True
    config_bi = ConfigBI()

    bi = BinocularIntegration(engine, config_bi)
    bi.config.batch_size = 2
    bi.config.batch_size_val = 2

    # prepare dataset
    train_loader, validation_loader, test_loader = engine.prepare_dataset()

    # compute integration ratio and count
    bi.compute_bino_integration(train_loader)
    bi.plotLine_integration("ratio")
    bi.plotLine_integration("count")


# %%
if __name__ == "__main__":
    main()
