# %% load necessary modules
import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn import svm

from joblib import Parallel, delayed
from jaxtyping import Float


# %%
def load_train_data(file_dir: str, background_flag: bool = 1):
    """
        load cRDS predicted disparity map for training dataset in the
        classification task using SVM

    Args:
        file_dir ([string]): the directory location containing the data
            file_dir = f"{save_dir}/epoch_{epoch_to_load}/SVM_analysis"

        background_flag (binary): flag indicating simulating RDS with or without
                cRDS background

            0 -> without cRDS background
            1 -> with cRDS background

    Returns:
        X_train [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp, w] :
            predicted disparity map for aRDS
        Y_train [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp] :
            disparity labels for aRDS

        x_mean_all_dotDens ([n_dotDens, 1, w]]): mean of training dataset
        x_std_all_dotDens ([n_dotDens, 1, w]): std of training dataset
    """
    if background_flag:  # with cRDS background
        # load predicted disparity data for crds
        disp_map = np.load(f"{file_dir}/pred_disp_crds.npy")
        # disp_map = np.load(f"{rdsa.xDecode_dir}/pred_disp_crds.npy")
        # load disparity labels for crds
        Y_train = np.load(f"{file_dir}/pred_disp_labels_crds.npy")
        # Y_train = np.load(f"{rdsa.xDecode_dir}/pred_disp_labels_crds.npy")

    else:  # without cRDS background
        # load predicted disparity data for crds
        disp_map = np.load(f"{file_dir}/pred_disp_crds_wo_bg.npy")
        # load disparity labels for crds
        Y_train = np.load(f"{file_dir}/pred_disp_labels_crds_wo_bg.npy")

    # average across rows, [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp, h, w] =>
    # [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp, w]
    disp_map_avg = disp_map.mean(axis=-2)

    # compute mean and std for normalization across batch, for each dot density
    x_mean_all_dotDens = disp_map_avg.mean(axis=1, keepdims=True)  # [n_dotDens, 1, w]
    x_std_all_dotDens = disp_map_avg.std(axis=1, keepdims=True)  # [n_dotDens, 1, w]
    # prevent division by zero
    x_std_all_dotDens[x_std_all_dotDens == 0] = 1e-6

    # standardize training data crds
    X_train = (disp_map_avg - x_mean_all_dotDens) / x_std_all_dotDens
    Y_train = Y_train.reshape(disp_map_avg.shape[0], -1)

    return X_train, Y_train, x_mean_all_dotDens, x_std_all_dotDens


def load_test_data(
    file_dir: str,
    x_mean_all_dotDens: Float[np.ndarray, "n_dotDens 1 w"],
    x_std_all_dotDens: Float[np.ndarray, "n_dotDens 1 w"],
    background_flag: bool = 1,
):
    """
    load aRDS and hmRDS predicted disparity map for test dataset in
    classification task using SVM

    Args:
        file_dir ([string]): the directory location containing the data
            file_dir = f"{save_dir}/epoch_{epoch_to_load}/SVM_analysis"

        x_mean_all_dotDens ([n_dotDens, 1, w]]): mean of training dataset
        x_std_all_dotDens ([n_dotDens, 1, w]): std of training dataset

        background_flag (binary): flag indicating simulating RDS with or without
                cRDS background

            0 -> without cRDS background
            1 -> with cRDS background

    Returns:
        X_ards [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp, h, w] :
            predicted disparity map for aRDS

        Y_ards [len(dotDens_list), 2 * n_rds_each_disp] : disparity labels for aRDS

        X_hmrds [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp, h, w] :
            predicted disparity map for hmRDS

        Y_hmrds [len(dotDens_list), 2 * n_rds_each_disp]: disparity labels for hmRDS
    """

    ## aRDS
    if background_flag:  # with cRDS background
        ## load predicted disparity data for ards
        disp_map = np.load(f"{file_dir}/pred_disp_ards.npy")
        # disp_map = np.load(f"{rdsa.xDecode_dir}/pred_disp_ards.npy")
        # load disparity labels for ards
        Y_ards = np.load(f"{file_dir}/pred_disp_labels_ards.npy")
    else:  # without cRDS background
        ## load predicted disparity data for ards
        disp_map = np.load(f"{file_dir}/pred_disp_ards_wo_bg.npy")
        # load disparity labels for ards
        Y_ards = np.load(f"{file_dir}/pred_disp_labels_ards_wo_bg.npy")

    # average across rows
    disp_map_avg = disp_map.mean(
        axis=-2
    )  # [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp, w]

    # standardize data
    X_ards = (disp_map_avg - x_mean_all_dotDens) / x_std_all_dotDens
    n_dotDens = disp_map_avg.shape[0]
    Y_ards = Y_ards.reshape(n_dotDens, -1)

    ## hmRDS
    if background_flag:  # with cRDS background
        ## load predicted disparity data for ards
        disp_map = np.load(f"{file_dir}/pred_disp_hmrds.npy")
        # load disparity labels for ards
        Y_hmrds = np.load(f"{file_dir}/pred_disp_labels_hmrds.npy")
    else:  # without cRDS background
        ## load predicted disparity data for ards
        disp_map = np.load(f"{file_dir}/pred_disp_hmrds_wo_bg.npy")
        # load disparity labels for ards
        Y_hmrds = np.load(f"{file_dir}/pred_disp_labels_hmrds_wo_bg.npy")

    # average across rows
    disp_map_avg = disp_map.mean(
        axis=-2
    )  # [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp, w]

    # normalize data
    X_hmrds = (disp_map_avg - x_mean_all_dotDens) / x_std_all_dotDens
    Y_hmrds = Y_hmrds.reshape(n_dotDens, -1)

    return X_ards, Y_ards, X_hmrds, Y_hmrds


# %%
def xDecode_single_bootstrap(
    X_train: Float[np.ndarray, "n_dotDens n_rds w"],
    Y_train: Float[np.ndarray, "n_dotDens n_rds"],
    X_ards: Float[np.ndarray, "n_dotDens n_rds w"],
    Y_ards: Float[np.ndarray, "n_dotDens n_rds"],
    X_hmrds: Float[np.ndarray, "n_dotDens n_rds w"],
    Y_hmrds: Float[np.ndarray, "n_dotDens n_rds"],
    split_train_ratio: float,
    dotDens_list: list,
    iter_bootstrap: int,
    n_bootstrap: int,
):
    n_dotDens = len(dotDens_list)
    n_samples = X_train.shape[1]
    n_train = int(split_train_ratio * n_samples)
    score_ards = np.empty(n_dotDens, dtype=np.float32)
    score_hmrds = np.empty(n_dotDens, dtype=np.float32)
    score_crds = np.empty(n_dotDens, dtype=np.float32)
    predict_ards = np.empty((n_dotDens, n_samples), dtype=np.int8)
    predict_hmrds = np.empty((n_dotDens, n_samples), dtype=np.int8)
    predict_crds = np.empty((n_dotDens, n_samples - n_train), dtype=np.int8)

    # generate random numbers for splitting train and test dataset
    idx = np.random.permutation(n_samples)
    idx_train = idx[:n_train]
    idx_test = idx[n_train:]
    for dd in range(n_dotDens):
        print(
            f"{iter_bootstrap+1}/{n_bootstrap} SVM on RDS with dotDens: {dotDens_list[dd]:.1f}"
        )

        # split training and test data
        X_train_split = X_train[dd, idx_train]
        Y_train_split = Y_train[dd, idx_train]
        X_test = X_train[dd, idx_test]
        Y_test = Y_train[dd, idx_test]

        # train classifier
        clf = svm.SVC(kernel="linear", cache_size=1000)
        clf.fit(X_train_split, Y_train_split)

        ## evaluate on ards
        # predict output
        predict_ards[dd] = clf.predict(X_ards[dd])
        # compute score
        score_ards[dd] = clf.score(X_ards[dd], Y_ards[dd])

        ## evaluate on hmrds
        # predict output
        predict_hmrds[dd] = clf.predict(X_hmrds[dd])
        # compute score
        score_hmrds[dd] = clf.score(X_hmrds[dd], Y_hmrds[dd])

        ## evaluate crds test subset
        # predict output
        predict_crds[dd] = clf.predict(X_test)
        # compute score
        score_crds[dd] = clf.score(X_test, Y_test)

        # clean up classifier
        del clf

    return (
        score_ards,
        score_hmrds,
        score_crds,
        predict_ards,
        predict_hmrds,
        predict_crds,
    )


# %%
def xDecode_bootstrap(
    X_train: Float[np.ndarray, "n_dotDens n_rds w"],
    Y_train: Float[np.ndarray, "n_dotDens n_rds"],
    X_ards: Float[np.ndarray, "n_dotDens n_rds w"],
    Y_ards: Float[np.ndarray, "n_dotDens n_rds"],
    X_hmrds: Float[np.ndarray, "n_dotDens n_rds w"],
    Y_hmrds: Float[np.ndarray, "n_dotDens n_rds"],
    split_train_ratio: float,
    n_bootstrap: int,
    dotDens_list: list,
):

    output = Parallel(n_jobs=-1)(
        delayed(xDecode_single_bootstrap)(
            X_train,
            Y_train,
            X_ards,
            Y_ards,
            X_hmrds,
            Y_hmrds,
            split_train_ratio,
            dotDens_list,
            iter_bootstrap,
            n_bootstrap,
        )
        for iter_bootstrap in range(n_bootstrap)
    )

    # unpack output
    score_ards_bootstrap = np.array(
        [result[0] for result in output]
    )  # [n_bootstrap, n_dotDens]
    score_hmrds_bootstrap = np.array([result[1] for result in output])
    score_crds_bootstrap = np.array([result[2] for result in output])
    predict_ards_bootstrap = np.array([result[3] for result in output])
    predict_hmrds_bootstrap = np.array([result[4] for result in output])
    predict_crds_bootstrap = np.array([result[5] for result in output])

    return (
        score_ards_bootstrap,
        predict_ards_bootstrap,
        score_hmrds_bootstrap,
        predict_hmrds_bootstrap,
        score_crds_bootstrap,
        predict_crds_bootstrap,
    )
