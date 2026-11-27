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
    # train_loader, validation_loader, test_loader = engine.prepare_dataset()

    # compute monocular and binocular responses, spatially
    # average along Height and Width dimension
    # bi.compute_mono_and_bino_resp_spatial_avg(train_loader)

    # compute binocular summation and binocular multiplication, spatially
    # average along Height and Width dimension
    # bi.compute_bino_sum_and_bino_mul_spatial_avg(train_loader)
    bi.plotLine_bino_correlation_interaction()

    # plot scatter mono_left_resp vs mono_right_resp,
    # bino_mul_resp vs bino_resp,
    # bino_sum_resp vs bino_resp
    bi.plotScatter_bino_correlation_interaction()

    # # compute integration ratio and count
    # bi.compute_bino_integration(train_loader)
    # bi.plotLine_bino_integration_interaction("ratio")

    # plot integration averaged across seeds for all interactions
    # bi.plotLine_bino_integration_all_interactions("ratio")


# %%
if __name__ == "__main__":
    main()
