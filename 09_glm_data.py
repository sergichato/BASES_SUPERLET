import numpy as np
import pandas as pd

from help_funcs import load_exp_variables


# ----- #
# Hyperparameters.
metric = "rocauc"
metric_str = "score"

band = "beta"

limit_hspace = True
if limit_hspace == True:
    hspace_str = "_sel"
elif limit_hspace == False:
    hspace_str = ""

remove_fooof = False
if remove_fooof == True:
    fooof_save_str = ""
elif remove_fooof == False:
    fooof_save_str = "_nfs"


# ----- #
# Dataset selection.
datas = [
    "zhou2016",
    "2014004",
    "2014001",
    "munichmi",
    "weibo2014",
    "cho2017",
]

# Mode.
mode = "local"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"


# ----- #
# Data parsing.
all_datasets_1 = []
all_subjects_1 = []
all_variables_1 = []

all_datasets_2 = []
all_subjects_2 = []
all_variables_2 = []

results_bw_p = []
results_ball = []

all_trials_1 = []
all_trials_2 = []


for d, data in enumerate(datas):
    if data == "zhou2016":
        variables_path = "{}zhou_2016/variables.json".format(basepath)
        title_str = "Zhou 2016"
    elif data == "2014004":
        variables_path = "{}2014_004/variables.json".format(basepath)
        title_str = "BNCI 2014-004"
    elif data == "2014001":
        variables_path = "{}2014_001/variables.json".format(basepath)
        title_str = "BNCI 2014-001"
    elif data == "munichmi":
        variables_path = "{}munichmi/variables.json".format(basepath)
        title_str = "Munich MI (Grosse-Wentrup)"
    elif data == "cho2017":
        variables_path = "{}cho_2017/variables.json".format(basepath)
        title_str = "Cho 2017"
    elif data == "weibo2014":
        variables_path = "{}weibo_2014/variables.json".format(basepath)
        title_str = "Weibo 2014"

    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    savepath = experimental_vars["dataset_path"]

    subs = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
    if data == "cho2017":
        # Some subjects are not included in the dataset.
        subs = np.delete(np.array(subs), [31, 45, 48]).tolist()

    # Dataset id.
    all_datasets_1.append(np.repeat(d + 1, len(subs) * 2))

    # Subject id.
    all_subjects_1.append(np.tile(subs, 2))

    # Feature id.
    all_variables_1.append(
        np.hstack((np.repeat(1, len(subs)), np.repeat(2, len(subs))))
    )

    # Dataset id.
    all_datasets_2.append(np.repeat(d + 1, len(subs) * 5))

    # Subject id.
    all_subjects_2.append(np.tile(subs, 5))

    # Feature id.
    all_variables_2.append(
        np.hstack(
            (
                np.repeat(1, len(subs)),
                np.repeat(2, len(subs)),
                np.repeat(3, len(subs)),
                np.repeat(4, len(subs)),
                np.repeat(5, len(subs)),
            )
        )
    )

    # ----- #
    # Loading of decoding results.
    results_bw = np.load(
        savepath
        + "mean_{}_stratified_{}_bursts_pca{}{}.npy".format(
            metric, band, fooof_save_str, hspace_str
        )
    )
    results_br = np.load(
        savepath + "mean_{}_{}_rate_simple{}.npy".format(metric, band, fooof_save_str)
    )
    results_bf = np.load(
        savepath + "mean_{}_{}{}_tf_features.npy".format(metric, band, fooof_save_str)
    )
    results_ba = np.load(
        savepath + "mean_{}_{}{}_tf_features.npy".format(metric, band, fooof_save_str)
    )
    results_bv = np.load(
        savepath + "mean_{}_{}{}_tf_volume.npy".format(metric, band, fooof_save_str)
    )
    results_p = np.load(
        savepath
        + "mean_{}_power_{}_band_simple{}.npy".format(metric, band, fooof_save_str)
    )

    # All subjects' results.
    n_trials = []
    res_bw = []
    res_br = []
    res_bf = []
    res_ba = []
    res_bv = []
    res_p = []

    for s, subj in enumerate(subs):
        # Corresponding classification results.
        res_bw.append(np.nanmean(results_bw, axis=1)[s])
        res_br.append(np.nanmean(results_br, axis=(1, 2))[s])
        res_bf.append(np.nanmean(results_bf, axis=1)[s])
        res_ba.append(np.nanmean(results_ba, axis=1)[s])
        res_bv.append(np.nanmean(results_bv, axis=1)[s])
        res_p.append(np.nanmean(results_p, axis=(1, 2))[s])

        sub_trials = np.load(savepath + "sub_{}/ntrials.npy".format(subj))
        n_trials.append(sub_trials[0, 2] + sub_trials[0, 3])

    results_bw_p.append(np.hstack((res_bw, res_p)))
    results_ball.append(np.hstack((res_bw, res_br, res_bf, res_ba, res_bv)))

    all_trials_1.append(np.tile(n_trials, 2))
    all_trials_2.append(np.tile(n_trials, 5))


# ----- #
# GLMs
all_datasets_pd_1 = np.hstack(all_datasets_1)
all_subjects_pd_1 = np.hstack(all_subjects_1)
all_variables_pd_1 = np.hstack(all_variables_1)
all_trials_pd_1 = np.hstack(all_trials_1)
all_results_pd_1 = np.hstack(results_bw_p)

all_datasets_pd_2 = np.hstack(all_datasets_2)
all_subjects_pd_2 = np.hstack(all_subjects_2)
all_variables_pd_2 = np.hstack(all_variables_2)
all_trials_pd_2 = np.hstack(all_trials_2)
all_results_pd_2 = np.hstack(results_ball)

# Full model as pandas dataframe.
lmm_data_1 = pd.DataFrame(
    np.array(
        [
            all_datasets_pd_1,
            all_subjects_pd_1,
            all_variables_pd_1,
            all_trials_pd_1,
            all_results_pd_1,
        ]
    ).T,
    columns=["Dataset", "Subject", "Feature", "Trials", "Accuracy"],
)

lmm_data_2 = pd.DataFrame(
    np.array(
        [
            all_datasets_pd_2,
            all_subjects_pd_2,
            all_variables_pd_2,
            all_trials_pd_2,
            all_results_pd_2,
        ]
    ).T,
    columns=["Dataset", "Subject", "Feature", "Trials", "Accuracy"],
)


savepath = "/home/sotpapad/Codes/bebop_test/"

lmm_data_1.to_csv(savepath + "class_res_1{}{}.csv".format(fooof_save_str, hspace_str))
lmm_data_2.to_csv(savepath + "class_res_2{}{}.csv".format(fooof_save_str, hspace_str))
