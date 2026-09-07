# %% load necessary modules
import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()
from sklearn import svm

from joblib import Parallel, delayed


# %%
def load_train_data(file_dir, background_flag=1):
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
        X_train [len(dotDens_list), len(disp_ct_pix) * n_rds_each_disp, h, w] :
            predicted disparity map for aRDS
        Y_train [len(dotDens_list), 2 * n_rds_each_disp] : disparity labels for aRDS

        x_mean ([float32]]): mean of training dataset
        x_std ([float32]): std of training dataset
    """
    if background_flag:  # with cRDS background
        # load predicted disparity data for crds
        disp_map = np.load(f"{file_dir}/pred_disp_crds.npy")
        # load disparity labels for crds
        Y_train = np.load(f"{file_dir}/pred_disp_labels_crds.npy")

    else:  # without cRDS background
        # load predicted disparity data for crds
        disp_map = np.load(f"{file_dir}/pred_disp_crds_wo_bg.npy")
        # load disparity labels for crds
        Y_train = np.load(f"{file_dir}/pred_disp_labels_crds_wo_bg.npy")

    # compute mean and std for normalization
    x_mean = disp_map.mean()  # mean of training dataset
    x_std = disp_map.std()  # std of training dataset

    # normalize and reshape data
    n_dotDens, n_train, _, _ = disp_map.shape
    X_train = []
    for dd in range(n_dotDens):
        for i in range(n_train):
            # if Y_train[dd, i] in label_to_train:
            disp_map_this = (disp_map[dd, i] - x_mean) / x_std
            X_train.append(disp_map_this.flatten())
            # Y_train2.append(Y_train[dd, i])
    X_train = np.array(X_train)
    Y_train = Y_train.reshape(-1)

    return X_train, Y_train, x_mean, x_std


def load_test_data(file_dir, x_mean, x_std, background_flag=1):
    """
    load aRDS and hmRDS predicted disparity map for test dataset in
    classification task using SVM

    Args:
        file_dir ([string]): the directory location containing the data
            file_dir = f"{save_dir}/epoch_{epoch_to_load}/SVM_analysis"

        x_mean ([float32]]): mean of training dataset
        x_std ([float32]): std of training dataset

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
    if background_flag:  # with cRDS background
        ## load predicted disparity data for ards
        disp_map = np.load(f"{file_dir}/pred_disp_ards.npy")
        # load disparity labels for ards
        Y_ards = np.load(f"{file_dir}/pred_disp_labels_ards.npy")
    else:  # without cRDS background
        ## load predicted disparity data for ards
        disp_map = np.load(f"{file_dir}/pred_disp_ards_wo_bg.npy")
        # load disparity labels for ards
        Y_ards = np.load(f"{file_dir}/pred_disp_labels_ards_wo_bg.npy")

    # normalize data
    X_ards = (disp_map - x_mean) / x_std

    # reshape
    X_ards = X_ards.reshape([X_ards.shape[0], X_ards.shape[1], -1])

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

    # normalize data
    X_hmrds = (disp_map - x_mean) / x_std

    # reshape
    X_hmrds = X_hmrds.reshape([X_hmrds.shape[0], X_hmrds.shape[1], -1])

    return X_ards, Y_ards, X_hmrds, Y_hmrds


# %% resampling training data
def split_train_data(X_train, Y_train, n_samples):
    """
    resampling training data for bootstrap

    Args:
        X_train ([n_dotDens * len(disp_ct) * n_rds_each_disp, h * w], np.float32):
            training data

            loaded from:
            disp_map = np.load(f"{save_dir}/epoch_{epoch_to_load}/SVM_analysis/pred_disp_ards.npy")


        Y_train ([n_dotDens * len(disp_ct) * n_rds_each_disp], np.int8):
            disparity labels

        n_samples ([int8]): the size of resampled data

    Returns:
        X_resampled([n_samples, h * w], np.float32):
            resampled_data

        Y_resampled ([n_samples], np.int8):
            disparity labels of the resampled data
    """

    n_train = X_train.shape[0]

    id_resampled = np.random.choice(n_train, size=n_samples, replace=False)
    X_train_split = X_train[id_resampled]
    Y_train_split = Y_train[id_resampled]
    X_test_split = X_train[np.setdiff1d(np.arange(n_train), id_resampled)]
    Y_test_split = Y_train[np.setdiff1d(np.arange(n_train), id_resampled)]

    return X_train_split, Y_train_split, X_test_split, Y_test_split


# %%
def xDecode_single_bootstrap(
    X_train,
    Y_train,
    X_ards,
    Y_ards,
    X_hmrds,
    Y_hmrds,
    n_samples,
    dotDens_list,
    iter_bootstrap,
    n_bootstrap,
):
    n_dotDens = len(dotDens_list)
    n_rds = X_train.shape[0] // n_dotDens
    # n_crds = n_rds - (n_samples // n_dotDens)  # only use the split train-test
    n_crds = X_train.shape[0] - n_samples  # only use the split train-test
    score_ards = np.zeros(n_dotDens, dtype=np.float32)
    score_hmrds = np.zeros(n_dotDens, dtype=np.float32)
    score_crds = np.zeros(n_dotDens, dtype=np.float32)
    predict_ards = np.zeros((n_dotDens, n_rds), dtype=np.int8)
    predict_hmrds = np.zeros((n_dotDens, n_rds), dtype=np.int8)
    predict_crds = np.zeros((n_dotDens, n_crds), dtype=np.int8)

    # resample data
    # n_samples = 8000
    X_train_split, Y_train_split, X_test_split, Y_test_split = split_train_data(
        X_train, Y_train, n_samples
    )

    # define classifier
    clf = svm.SVC(kernel="linear", cache_size=1000)
    clf.fit(X_train_split, Y_train_split)

    for dd in range(n_dotDens):
        print(
            f"{iter_bootstrap+1}/{n_bootstrap} SVM on RDS with dotDens: {dotDens_list[dd]:.1f}"
        )

        ## ards
        # predict output
        predict_ards[dd] = clf.predict(X_ards[dd])

        # compute score
        score_ards[dd] = clf.score(X_ards[dd], Y_ards[dd])

        ## hmrds
        # predict output
        predict_hmrds[dd] = clf.predict(X_hmrds[dd])

        # compute score
        score_hmrds[dd] = clf.score(X_hmrds[dd], Y_hmrds[dd])

        ## crds
        # predict output
        predict_crds[dd] = clf.predict(X_test_split)

        # compute score
        score_crds[dd] = clf.score(X_test_split, Y_test_split)

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
    X_train,
    Y_train,
    X_ards,
    Y_ards,
    X_hmrds,
    Y_hmrds,
    n_samples,
    n_bootstrap,
    dotDens_list,
):
    n_dotDens = len(dotDens_list)
    output = []
    output.append(
        Parallel(n_jobs=2)(
            delayed(xDecode_single_bootstrap)(
                X_train,
                Y_train,
                X_ards,
                Y_ards,
                X_hmrds,
                Y_hmrds,
                n_samples,
                dotDens_list,
                iter_bootstrap,
                n_bootstrap,
            )
            for iter_bootstrap in range(n_bootstrap)
        )
    )

    # unpack
    n_dotDens = len(dotDens_list)
    n_rds = X_train.shape[0] // n_dotDens
    # n_crds = n_rds - (n_samples // n_dotDens)  # only use the split train-test
    n_crds = X_train.shape[0] - n_samples  # only use the split train-test
    score_ards_bootstrap = np.zeros((n_bootstrap, n_dotDens), dtype=np.float32)
    score_hmrds_bootstrap = np.zeros((n_bootstrap, n_dotDens), dtype=np.float32)
    score_crds_bootstrap = np.zeros((n_bootstrap, n_dotDens), dtype=np.float32)
    predict_ards_bootstrap = np.zeros((n_bootstrap, n_dotDens, n_rds), dtype=np.int8)
    predict_hmrds_bootstrap = np.zeros((n_bootstrap, n_dotDens, n_rds), dtype=np.int8)
    predict_crds_bootstrap = np.zeros((n_bootstrap, n_dotDens, n_crds), dtype=np.int8)
    for i in range(n_bootstrap):
        # score_ards
        temp = output[0][i][0]
        score_ards_bootstrap[i] = temp

        # score_hmrds
        temp = output[0][i][1]
        score_hmrds_bootstrap[i] = temp

        # score_crds
        temp = output[0][i][2]
        score_crds_bootstrap[i] = temp

        # predict_ards
        temp = output[0][i][3]
        predict_ards_bootstrap[i] = temp

        # predict_hmrds
        temp = output[0][i][4]
        predict_hmrds_bootstrap[i] = temp

        # predict_crds
        temp = output[0][i][5]
        predict_crds_bootstrap[i] = temp

    return (
        score_ards_bootstrap,
        predict_ards_bootstrap,
        score_hmrds_bootstrap,
        predict_hmrds_bootstrap,
        score_crds_bootstrap,
        predict_crds_bootstrap,
    )


# %%
