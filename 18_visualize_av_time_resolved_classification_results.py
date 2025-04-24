import numpy as np
import matplotlib.pyplot as plt
import pickle

from os.path import join
from matplotlib.style import use

from preprocess import (
    load_sub,
    apply_preprocessing,
)
from help_funcs import load_exp_variables, vis_permutation_cluster_test

use("default")


# ----- #
def itr_tr(score, time, classification_mode, bin_dt):

    # Binary entropy.
    H = -score * np.log2(score) - (1 - score) * np.log2(1 - score)

    # A perfect score of 1 results in NAN, so set it to 0.
    H[np.where(np.isnan(H))] = 0

    # Time shift to only use positive values.
    if classification_mode == "incremental":
        new_time = [t - time[0] + bin_dt if t <= 0 else t for t in time]
        new_time = np.around(new_time, 3)
    elif classification_mode == "sliding":
        new_time = time - time[0] + bin_dt

    # ITR.
    if H.shape[-1] == new_time.shape[0]:
        itr = (1 - H) / new_time
    else:
        temp = np.rollaxis(H, 3, 2)
        itr_temp = (1 - temp) / new_time
        itr = np.rollaxis(itr_temp, 3, 2)

    return itr


# ----- #
# Hyperparameters.
remove_fooof = False
if remove_fooof == True:
    fooof_save_str = ""
elif remove_fooof == False:
    fooof_save_str = "_nfs"

metric = "rocauc"
if metric == "rocauc":
    metric_str = "score"
elif metric == "accuracy":
    metric_str = metric

classification_mode = "incremental"  # "incremental", "sliding"

perm = True         # True, False
n_perm = 2**13
threshold = 0.05        # <float, e.g. 0.05>, None
correction = None       # "bonferroni", "sidak", None

# Visualize ITR instead of decoding score?
acc_time_index = False   # False, True
ati_zoom = False        # False, True

sl_time_center = "start"  # "start", "mid", "end"

savefigs = True  # True, False

plot_format = "pdf"  # "pdf", "png"


# ----- #
# Dataset selection.
datas = [
    "2014001",
    "2014004",
    "cho2017",
    "dreyer2023",
    "munichmi",
    "weibo2014",
    "zhou2016",
]

# Mode.
mode = "local"  # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/"


# ----- #
# Results selection.
clm_str_tr = "_tr"
clm_str_sl = "_sl"

if classification_mode == "incremental":
    clm_str = clm_str_tr
    # HARD CODED ASSUMPTION: convolution is
    # always the first of either category.
    # """
    res_to_vis = [
        "conv",
        "fb_1_beta",
        "fb_1_mubeta",
        "fb_beta",
        "fb_mubeta",
    ]
    # """
    """
    res_to_vis = [
        "conv",
        "fb_1_mu",
        "fb_mu",
    ]
    """
elif classification_mode == "sliding":
    clm_str = clm_str_sl
    # HARD CODED ASSUMPTION: convolution is
    # always the first of either category.
    # """
    res_to_vis = [
        "conv_sliding",
        "fb_1_beta_sliding",
        "fb_1_mubeta_sliding",
        "fb_beta_sliding",
        "fb_mubeta_sliding",
    ]
    # """
    """
    res_to_vis = [
        "conv_sliding",
        "fb_1_mu_sliding",
        "fb_mu_sliding",
    ]
    """

filter_banks = [
    [[6, 9], [9, 12], [12, 15]],
    [[15, 18], [18, 21], [21, 24], [24, 27], [27, 30]],
    [[6, 9], [9, 12], [12, 15], [15, 18], [18, 21], [21, 24], [24, 27], [27, 30]],
    [[6, 15]],
    [[15, 30]],
    [[6, 30]],
]

power_band = filter_banks[3][0]


# ----- #
# Figure initialization.
screen_res = [1920, 972]
dpi = 300

linewidths = [1.5, 0.75, 1.0]
fontsizes = [6, 5, 4]
tick_size = 4

if savefigs == False:
    hratios_0 = np.ones(len(datas) + 1)

    fig0 = plt.figure(
        constrained_layout=False,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
    gs0 = fig0.add_gridspec(
        nrows=len(datas) + 1,
        ncols=4,
        wspace=0.25,
        hspace=0.5,
        left=0.05,
        right=0.95,
        top=0.90,
        bottom=0.05,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        height_ratios=hratios_0,
    )

else:
    hratios_0 = np.ones(len(datas) + 1)

    wide = 4.5

    fig0 = plt.figure(
        constrained_layout=False, figsize=(wide, 1.5 * len(datas)), dpi=dpi
    )
    gs0 = fig0.add_gridspec(
        nrows=len(datas) + 1,
        ncols=4,
        wspace=0.25,
        hspace=0.40,
        left=0.10,
        right=0.98,
        top=0.90,
        bottom=0.10,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
        height_ratios=hratios_0,
    )


# ----- #
# Data loading.
across_datasets = {}
across_datasets_std = {}
pipe_ids_all = {
    "filter": [],
    "filter bank": [],
}

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
        title_str = "Munich MI\n(Grosse-Wentrup 2009)"
    elif data == "cho2017":
        variables_path = "{}cho_2017/variables.json".format(basepath)
        title_str = "Cho 2017"
    elif data == "weibo2014":
        variables_path = "{}weibo_2014/variables.json".format(basepath)
        title_str = "Weibo 2014"
    elif data == "dreyer2023":
        variables_path = "{}dreyer_2023/variables.json".format(basepath)
        title_str = "Dreyer 2023"

    print("Analyzing {} dataset...".format(title_str))

    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    savepath = experimental_vars["dataset_path"]

    subs = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
    if data == "cho2017":
        # Some subjects are not included in the dataset.
        subs = np.delete(np.array(subs), [31, 45, 48]).tolist()

    exp_time_periods = experimental_vars["exp_time_periods"]

    bin_dt = experimental_vars["bin_dt"]
    tmin = experimental_vars["tmin"]
    tmax = experimental_vars["tmax"]
    sfreq = experimental_vars["sfreq"]

    # Time in experiment.
    exp_time = np.linspace(tmin, tmax, int((np.abs(tmax - tmin)) * sfreq) + 1)
    exp_time = np.around(exp_time, decimals=3)
    task_time_lims = [exp_time_periods[1], exp_time_periods[2]]
    base_start = -0.5

    samples_step = np.around(exp_time[1] - exp_time[0], decimals=4)
    window_samples = int(bin_dt / samples_step)

    baseline_window_end = np.where(exp_time <= task_time_lims[0] - 0.5)[0][-1]
    rebound_window_end = np.where(exp_time <= task_time_lims[1] + 0.5)[0][-1]
    n_windows = int((rebound_window_end - baseline_window_end) / window_samples) + 1

    window_length_sl = 1.0
    step_length = 0.05
    if classification_mode == "sliding":
        bin_dt = step_length
    window_samples_sl = window_length_sl / samples_step
    step_samples = step_length / samples_step
    n_windows_sl = int(
        np.floor(
            ((rebound_window_end - baseline_window_end) - window_samples_sl)
            / step_samples
        )
        + window_samples_sl / step_samples
        + 1
    )

    # Time vectors.
    # Time in 100ms non-overlapping windows.
    time = np.array(
        [exp_time[baseline_window_end + n * window_samples] for n in range(n_windows)]
    )

    # Time in 1000ms sliding (by 50ms) windows.
    time_sl = [
        exp_time[int(n * step_samples + step_samples)] for n in range(n_windows_sl)
    ]
    time_sl = np.array(time_sl)

    if sl_time_center == "mid":
        time_sl = [t + 0.5 for t in time_sl]
        time_sl = np.around(time_sl, 3)
    elif sl_time_center == "end":
        time_sl = [t + 1.0 for t in time_sl]

    # Shifted ITR time of sliding window technique.
    new_time_sl = time_sl - time_sl[0] + bin_dt

    # "Important" time points.
    time_points = [0.0, 3.0, exp_time_periods[2]]
    if data != "cho2017" and data != "dreyer2023":
        time_ids = [np.where(time == tp)[0][0] for tp in time_points]
        time_ids_sl = [np.where(time_sl == tp)[0][0] for tp in time_points]
        equal_end = time_ids[1] + 1 if classification_mode == "incremental" else time_ids_sl[1] + 2
    else:
        time_ids = [np.where(time >= tp)[0][0] for tp in time_points]
        time_ids_sl = [np.where(time_sl >= tp)[0][0] for tp in time_points]
        equal_end = time_ids[1] if classification_mode == "incremental" else time_ids_sl[1]
    

    # ----- #
    # Loading of decoding results.
    # Dataset results.
    results = []
    results_std = []
    perm_data = []
    label_strs = []
    colors = []


    # ----- #
    # Inremental window results.
    # Convolution results.
    if "conv" in res_to_vis:
        label_strs.append("Convolution & CSP")
        colors.append("orangered")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_beta_band_csp{}_conv_waves{}.npy".format(
                        metric, fooof_save_str, clm_str_tr
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_beta_band_csp{}_conv_waves{}.npy".format(
                    metric, fooof_save_str, clm_str_tr
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_beta_band_csp{}_conv_waves{}.npy".format(
                        metric, fooof_save_str, clm_str_tr
                    )
                ),
                axis=-1,
            ),
        )

    # All filtering results.
    if "fb_1_beta" in res_to_vis:
        label_strs.append(
            "Beta band filter & CSP ({}-{} Hz)".format(
                filter_banks[4][0][0],
                filter_banks[4][0][1],
            ),
        )
        colors.append("goldenrod")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[4]),
                        filter_banks[4][0][0],
                        filter_banks[4][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[4]),
                    filter_banks[4][0][0],
                    filter_banks[4][-1][1],
                    clm_str_tr,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[4]),
                        filter_banks[4][0][0],
                        filter_banks[4][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_1_mu" in res_to_vis:
        label_strs.append(
            "Mu band filter & CSP ({}-{} Hz)".format(
                filter_banks[3][0][0],
                filter_banks[3][0][1],
            ),
        )
        colors.append("mediumturquoise")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[3]),
                        filter_banks[3][0][0],
                        filter_banks[3][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[3]),
                    filter_banks[3][0][0],
                    filter_banks[3][-1][1],
                    clm_str_tr,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[3]),
                        filter_banks[3][0][0],
                        filter_banks[3][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_1_mubeta" in res_to_vis:
        label_strs.append(
            "Mu-beta band filter & CSP ({}-{} Hz)".format(
                filter_banks[5][0][0],
                filter_banks[5][0][1],
            ),
        )
        colors.append("mediumorchid")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[5]),
                        filter_banks[5][0][0],
                        filter_banks[5][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[5]),
                    filter_banks[5][0][0],
                    filter_banks[5][-1][1],
                    clm_str_tr,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[5]),
                        filter_banks[5][0][0],
                        filter_banks[5][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=-1,
            ),
        )

    # All filter bank results.
    if "fb_beta" in res_to_vis:
        label_strs.append(
            "Filter bank & CSP ({}-{} Hz; {} bands; {} Hz step)".format(
                filter_banks[1][0][0],
                filter_banks[1][-1][1],
                len(filter_banks[1]),
                filter_banks[1][0][1] - filter_banks[1][0][0],
            ),
        )
        colors.append("goldenrod")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[1]),
                        filter_banks[1][0][0],
                        filter_banks[1][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[1]),
                    filter_banks[1][0][0],
                    filter_banks[1][-1][1],
                    clm_str_tr,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[1]),
                        filter_banks[1][0][0],
                        filter_banks[1][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_mu" in res_to_vis:
        label_strs.append(
            "Filter bank & CSP ({}-{} Hz; {} bands; {} Hz step)".format(
                filter_banks[0][0][0],
                filter_banks[0][-1][1],
                len(filter_banks[0]),
                filter_banks[0][0][1] - filter_banks[0][0][0],
            ),
        )
        colors.append("mediumturquoise")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[0]),
                        filter_banks[0][0][0],
                        filter_banks[0][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[0]),
                    filter_banks[0][0][0],
                    filter_banks[0][-1][1],
                    clm_str_tr,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[0]),
                        filter_banks[0][0][0],
                        filter_banks[0][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_mubeta" in res_to_vis:
        label_strs.append(
            "Filter bank & CSP ({}-{} Hz; {} bands; {} Hz step)".format(
                filter_banks[2][0][0],
                filter_banks[2][-1][1],
                len(filter_banks[2]),
                filter_banks[2][0][1] - filter_banks[2][0][0],
            ),
        )
        colors.append("mediumorchid")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[2]),
                        filter_banks[2][0][0],
                        filter_banks[2][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[2]),
                    filter_banks[2][0][0],
                    filter_banks[2][-1][1],
                    clm_str_tr,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[2]),
                        filter_banks[2][0][0],
                        filter_banks[2][-1][1],
                        clm_str_tr,
                    )
                ),
                axis=-1,
            ),
        )


    # ------ #
    # Sliding window results.
    # Convolution results.
    if "conv_sliding" in res_to_vis:
        label_strs.append("Convolution & CSP (slw)")
        colors.append("orangered")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_beta_band_csp{}_conv_waves{}.npy".format(
                        metric, fooof_save_str, clm_str_sl
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_beta_band_csp{}_conv_waves{}.npy".format(
                    metric, fooof_save_str, clm_str_sl
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_beta_band_csp{}_conv_waves{}.npy".format(
                        metric, fooof_save_str, clm_str_sl
                    )
                ),
                axis=-1,
            ),
        )

    # All filtering results.
    if "fb_1_beta_sliding" in res_to_vis:
        label_strs.append(
            "Beta band filter & CSP ({}-{} Hz / slw)".format(
                filter_banks[4][0][0],
                filter_banks[4][0][1],
            ),
        )
        colors.append("goldenrod")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[4]),
                        filter_banks[4][0][0],
                        filter_banks[4][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[4]),
                    filter_banks[4][0][0],
                    filter_banks[4][-1][1],
                    clm_str_sl,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[4]),
                        filter_banks[4][0][0],
                        filter_banks[4][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_1_mu_sliding" in res_to_vis:
        label_strs.append(
            "Mu band filter & CSP ({}-{} Hz / slw)".format(
                filter_banks[3][0][0],
                filter_banks[3][0][1],
            ),
        )
        colors.append("mediumturquoise")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[3]),
                        filter_banks[3][0][0],
                        filter_banks[3][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[3]),
                    filter_banks[3][0][0],
                    filter_banks[3][-1][1],
                    clm_str_sl,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[3]),
                        filter_banks[3][0][0],
                        filter_banks[3][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_1_mubeta_sliding" in res_to_vis:
        label_strs.append(
            "Mu-beta band filter & CSP ({}-{} Hz / slw)".format(
                filter_banks[5][0][0],
                filter_banks[5][0][1],
            ),
        )
        colors.append("mediumorchid")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[5]),
                        filter_banks[5][0][0],
                        filter_banks[5][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[5]),
                    filter_banks[5][0][0],
                    filter_banks[5][-1][1],
                    clm_str_sl,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[5]),
                        filter_banks[5][0][0],
                        filter_banks[5][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=-1,
            ),
        )

    # Filter bank results.
    if "fb_beta_sliding" in res_to_vis:
        label_strs.append(
            "Filter bank & CSP ({}-{} Hz; {} bands; {} Hz step / slw)".format(
                filter_banks[1][0][0],
                filter_banks[1][-1][1],
                len(filter_banks[1]),
                filter_banks[1][0][1] - filter_banks[1][0][0],
            ),
        )
        colors.append("goldenrod")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[1]),
                        filter_banks[1][0][0],
                        filter_banks[1][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[1]),
                    filter_banks[1][0][0],
                    filter_banks[1][-1][1],
                    clm_str_sl,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[1]),
                        filter_banks[1][0][0],
                        filter_banks[1][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_mu_sliding" in res_to_vis:
        label_strs.append(
            "Filter bank & CSP ({}-{} Hz; {} bands; {} Hz step / slw)".format(
                filter_banks[0][0][0],
                filter_banks[0][-1][1],
                len(filter_banks[0]),
                filter_banks[0][0][1] - filter_banks[0][0][0],
            ),
        )
        colors.append("mediumturquoise")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[0]),
                        filter_banks[0][0][0],
                        filter_banks[0][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[0]),
                    filter_banks[0][0][0],
                    filter_banks[0][-1][1],
                    clm_str_sl,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[0]),
                        filter_banks[0][0][0],
                        filter_banks[0][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_mubeta_sliding" in res_to_vis:
        label_strs.append(
            "Filter bank & CSP ({}-{} Hz; {} bands; {} Hz step / slw)".format(
                filter_banks[2][0][0],
                filter_banks[2][-1][1],
                len(filter_banks[2]),
                filter_banks[2][0][1] - filter_banks[2][0][0],
            ),
        )
        colors.append("mediumorchid")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[2]),
                        filter_banks[2][0][0],
                        filter_banks[2][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                    metric,
                    len(filter_banks[2]),
                    filter_banks[2][0][0],
                    filter_banks[2][-1][1],
                    clm_str_sl,
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                        metric,
                        len(filter_banks[2]),
                        filter_banks[2][0][0],
                        filter_banks[2][-1][1],
                        clm_str_sl,
                    )
                ),
                axis=-1,
            ),
        )


    # ----- #
    # Plots.

    # Across-subjects average temporal decoding.
    gs00 = gs0[d, 0:2].subgridspec(1, 1)
    ax00 = fig0.add_subplot(gs00[0])
    gs01 = gs0[d, 2:].subgridspec(1, 1)
    ax01 = fig0.add_subplot(gs01[0])

    tr_axes = [ax00, ax01]
    sl_axes = [ax00, ax01]

    if d == 0:
        gs02 = gs0[len(datas), 0:2].subgridspec(1, 1)
        ax02 = fig0.add_subplot(gs02[0])
        gs03 = gs0[len(datas), 2:].subgridspec(1, 1)
        ax03 = fig0.add_subplot(gs03[0])

        da_axes = [ax02, ax03]


    # ----- #
    # Permutation cluster tests.
    if acc_time_index == True:
        if classification_mode == "incremental":
            perm_data = itr_tr(np.array(perm_data), time, classification_mode, bin_dt)
        elif classification_mode == "sliding":
            perm_data = itr_tr(np.array(perm_data), time_sl, classification_mode, bin_dt)
    
    if (
        classification_mode == "incremental"
        and perm == True
        and "conv" in res_to_vis
        and len(res_to_vis) >= 2
    ):
        perm_cl_test = vis_permutation_cluster_test(
            perm_data,
            res_to_vis,
            colors,
            sub_id=None,
            n_perm=n_perm,
            threshold=threshold,
            correction=correction,
        )
    elif (
        classification_mode == "sliding"
        and perm == True
        and "conv_sliding" in res_to_vis
        and len(res_to_vis) >= 2
    ):
        perm_cl_test_sl = vis_permutation_cluster_test(
            perm_data,
            res_to_vis,
            colors,
            sub_id=None,
            n_perm=n_perm,
            threshold=threshold,
            correction=correction,
        )

    if acc_time_index == False:
        perm_offset_1 = 0.02
        if classification_mode == "sliding":
            perm_offset_0 = len(perm_cl_test_sl) / 100 + 0.01
        else:
            perm_offset_0 = len(perm_cl_test) / 100 + 0.01
    elif acc_time_index == True:
        if classification_mode == "sliding":
            perm_offset_1 = 0.62
            perm_offset_0 = len(perm_cl_test_sl) / 100 + 0.61
        else:
            perm_offset_1 = -0.26
            perm_offset_0 = len(perm_cl_test) / 100 - 0.27
    
    # Store all data for across-datasets permutations.
    if d == 0:
        perm_data_all = [pd[:, :equal_end] for pd in perm_data]
    else:
        for p, (pda, pd) in enumerate(zip(perm_data_all, perm_data)):
            perm_data_all[p] = np.vstack((pda, pd[:, :equal_end]))


    # ----- #
    # Fig 0: Time-resolved avergaged results.
    for r, (result, color, label_str) in enumerate(zip(results, colors, label_strs)):
        
        # Retrieve mean accuracy and std.
        across_subjects = np.mean(result, axis=0)
        across_subjects_std = np.std(result, axis=0) / np.sqrt(result.shape[0])

        # Convert to ITR if asked.
        if acc_time_index == True:
            if classification_mode == "incremental":
                across_subjects = np.mean(
                    [itr_tr(res, time, classification_mode, bin_dt) for res in result],
                    axis=0,
                )
                across_subjects_std = np.std(
                    [itr_tr(res, time, classification_mode, bin_dt) for res in result],
                    axis=0,
                ) / np.sqrt(result.shape[0])
            elif classification_mode == "sliding":
                across_subjects = np.mean(
                    [
                        itr_tr(res, time_sl, classification_mode, bin_dt)
                        for res in result
                    ],
                    axis=0,
                )
                across_subjects_std = np.std(
                    [
                        itr_tr(res, time_sl, classification_mode, bin_dt)
                        for res in result
                    ],
                    axis=0,
                ) / np.sqrt(result.shape[0])
        

        # ----- #
        # Plot across-subjects average accuracy.
        if label_str == "Convolution & CSP":
            # Time-resolved decoding accuracy.
            if d == 0:
                tr_axes[0].plot(
                    time,
                    across_subjects,
                    c=color,
                    linewidth=linewidths[0],
                    label=label_str,
                    zorder=3,
                )
                tr_axes[1].plot(time, across_subjects, c=color, linewidth=linewidths[0], zorder=3)
            else:
                tr_axes[0].plot(time, across_subjects, c=color, linewidth=linewidths[0], zorder=3)
                tr_axes[1].plot(time, across_subjects, c=color, linewidth=linewidths[0], zorder=3)

            # Across-subjects std.
            for ax0x in tr_axes:
                ax0x.fill_between(
                    time,
                    across_subjects - across_subjects_std,
                    across_subjects + across_subjects_std,
                    color=color,
                    alpha=0.2,
                    zorder=3,
                )
            
            # Across-datasets accuracy and std.
            if d == 0:
                across_datasets_std[label_str] = across_subjects_std[:equal_end]
                if acc_time_index == False:
                    across_datasets[label_str] = result[:, :equal_end]
                elif acc_time_index == True:
                    across_datasets[label_str] = itr_tr(result[:, :equal_end], time[:equal_end], classification_mode, bin_dt)
            else:
                across_datasets_std[label_str] = np.vstack(
                    (
                        across_datasets_std[label_str],
                        across_subjects_std[:equal_end] / np.sqrt(result.shape[0])
                    )
                )
                if acc_time_index == False:
                    across_datasets[label_str] = np.vstack((across_datasets[label_str], result[:, :equal_end]))
                elif acc_time_index == True:
                    across_datasets[label_str] = np.vstack(
                        (across_datasets[label_str],
                        itr_tr(result[:, :equal_end], time[:equal_end], classification_mode, bin_dt)
                        )
                    )
            
            if d == len(datas) - 1:
                for ax0x in da_axes:
                    ax0x.plot(
                        time[:equal_end],
                        np.mean(across_datasets[label_str], axis=0),
                        c=color,
                        linewidth=linewidths[0],
                        zorder=3,
                    )
                    if len(datas) <= 1:
                        ads = across_datasets_std[label_str]
                    else:
                        ads = np.mean(across_datasets_std[label_str], axis=0)
                    ax0x.fill_between(
                        time[:equal_end],
                        np.mean(across_datasets[label_str], axis=0) - ads,
                        np.mean(across_datasets[label_str], axis=0) + ads,
                        color=color,
                        alpha=0.2,
                        zorder=3,
                    )

        # Simple filters.
        if (
            label_str == "Beta band filter & CSP (15-30 Hz)"
            or label_str == "Mu band filter & CSP (6-15 Hz)"
            or label_str == "Mu-beta band filter & CSP (6-30 Hz)"
        ):
            # Time-resolved decoding accuracy.
            if d == 0:
                tr_axes[0].plot(
                    time,
                    across_subjects,
                    c=color,
                    linewidth=linewidths[0],
                    label=label_str,
                    zorder=2,
                )
            else:
                tr_axes[0].plot(time, across_subjects, c=color, linewidth=linewidths[0], zorder=2)

            # Across-subjects std.
            tr_axes[0].fill_between(
                time,
                across_subjects - across_subjects_std,
                across_subjects + across_subjects_std,
                color=color,
                alpha=0.2,
                zorder=2,
            )

            # Across-datasets accuracy and std.
            if d == 0:
                across_datasets_std[label_str] = across_subjects_std[:equal_end]
                if acc_time_index == False:
                    across_datasets[label_str] = result[:, :equal_end]
                elif acc_time_index == True:
                    across_datasets[label_str] = itr_tr(result[:, :equal_end], time[:equal_end], classification_mode, bin_dt)
            else:
                across_datasets_std[label_str] = np.vstack(
                    (
                        across_datasets_std[label_str],
                        across_subjects_std[:equal_end] / np.sqrt(result.shape[0])
                    )
                )
                if acc_time_index == False:
                    across_datasets[label_str] = np.vstack((across_datasets[label_str], result[:, :equal_end]))
                elif acc_time_index == True:
                    across_datasets[label_str] = np.vstack(
                        (across_datasets[label_str],
                        itr_tr(result[:, :equal_end], time[:equal_end], classification_mode, bin_dt)
                        )
                    )
            
            if d == len(datas) - 1:
                da_axes[0].plot(
                    time[:equal_end],
                    np.mean(across_datasets[label_str], axis=0),
                    c=color,
                    linewidth=linewidths[0],
                    zorder=2,
                )
                if len(datas) <= 1:
                    ads = across_datasets_std[label_str]
                else:
                    ads = np.mean(across_datasets_std[label_str], axis=0)
                da_axes[0].fill_between(
                    time[:equal_end],
                    np.mean(across_datasets[label_str], axis=0) - ads,
                    np.mean(across_datasets[label_str], axis=0) + ads,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

            # Permutation cluster test visualization.
            pipe_ids = np.where(np.array(label_strs) == label_str)[0]
            for pid in pipe_ids:
                dec_lines, dec_col = perm_cl_test[pid]
                tr_axes[0].plot(
                    time[time_ids[0] : time_ids[-1] + 1],
                    dec_lines[0][time_ids[0] : time_ids[-1] + 1] - perm_offset_1,
                    c=dec_col[0],
                    linewidth=linewidths[2],
                )
                tr_axes[0].plot(
                    time[time_ids[0] : time_ids[-1] + 1],
                    dec_lines[1][time_ids[0] : time_ids[-1] + 1] - perm_offset_1,
                    c=dec_col[1],
                    linewidth=linewidths[2],
                )
            
            # Global stats ids.
            if d == len(datas) - 1:
                pipe_ids_all["filter"].append(pipe_ids)

        # Filter banks.
        if (
            label_str == "Filter bank & CSP (15-30 Hz; 5 bands; 3 Hz step)"
            or label_str == "Filter bank & CSP (6-15 Hz; 3 bands; 3 Hz step)"
            or label_str == "Filter bank & CSP (6-30 Hz; 8 bands; 3 Hz step)"
        ):
            # Time-resolved decoding accuracy.
            tr_axes[1].plot(time, across_subjects, c=color, linewidth=linewidths[0], zorder=2)

            # Across-subjects std.
            tr_axes[1].fill_between(
                time,
                across_subjects - across_subjects_std,
                across_subjects + across_subjects_std,
                color=color,
                alpha=0.2,
                zorder=2,
            )

            # Across-datasets accuracy and std.
            if d == 0:
                across_datasets_std[label_str] = across_subjects_std[:equal_end]
                if acc_time_index == False:
                    across_datasets[label_str] = result[:, :equal_end]
                elif acc_time_index == True:
                    across_datasets[label_str] = itr_tr(result[:, :equal_end], time[:equal_end], classification_mode, bin_dt)
            else:
                across_datasets_std[label_str] = np.vstack(
                    (
                        across_datasets_std[label_str],
                        across_subjects_std[:equal_end] / np.sqrt(result.shape[0])
                    )
                )
                if acc_time_index == False:
                    across_datasets[label_str] = np.vstack((across_datasets[label_str], result[:, :equal_end]))
                elif acc_time_index == True:
                    across_datasets[label_str] = np.vstack(
                        (across_datasets[label_str],
                        itr_tr(result[:, :equal_end], time[:equal_end], classification_mode, bin_dt)
                        )
                    )
            
            if d == len(datas) - 1:
                da_axes[1].plot(
                    time[:equal_end],
                    np.mean(across_datasets[label_str], axis=0),
                    c=color,
                    linewidth=linewidths[0],
                    zorder=2,
                )
                if len(datas) <= 1:
                    ads = across_datasets_std[label_str]
                else:
                    ads = np.mean(across_datasets_std[label_str], axis=0)
                da_axes[1].fill_between(
                    time[:equal_end],
                    np.mean(across_datasets[label_str], axis=0) - ads,
                    np.mean(across_datasets[label_str], axis=0) + ads,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

            # Permutation cluster test visualization.
            pipe_ids = np.where(np.array(label_strs) == label_str)[0]
            for pid in pipe_ids:
                dec_lines, dec_col = perm_cl_test[pid]
                tr_axes[1].plot(
                    time[time_ids[0] : time_ids[-1] + 1],
                    dec_lines[0][time_ids[0] : time_ids[-1] + 1] - perm_offset_0,
                    c=dec_col[0],
                    linewidth=linewidths[2],
                )
                tr_axes[1].plot(
                    time[time_ids[0] : time_ids[-1] + 1],
                    dec_lines[1][time_ids[0] : time_ids[-1] + 1] - perm_offset_0,
                    c=dec_col[1],
                    linewidth=linewidths[2],
                )
            
            # Global stats ids.
            if d == len(datas) - 1:
                pipe_ids_all["filter bank"].append(pipe_ids)

        # Sliding window convolution decoding.
        if label_str == "Convolution & CSP (slw)":
            # Time-resolved decoding accuracy.
            sl_axes[0].plot(
                time_sl if acc_time_index == False else new_time_sl,
                across_subjects,
                c=color,
                linewidth=linewidths[0],
                label=label_str if d == 0 and classification_mode == "sliding" else "",
                zorder=3,
            )
            sl_axes[1].plot(
                time_sl if acc_time_index == False else new_time_sl,
                across_subjects,
                c=color,
                linewidth=linewidths[0],
                zorder=3,
            )

            # Across-subjects std.
            for ax0x in sl_axes:
                ax0x.fill_between(
                    time_sl if acc_time_index == False else new_time_sl,
                    across_subjects - across_subjects_std,
                    across_subjects + across_subjects_std,
                    color=color,
                    alpha=0.2,
                    zorder=3,
                )
            
           # Across-datasets accuracy and std.
            if d == 0:
                across_datasets_std[label_str] = across_subjects_std[:equal_end]
                if acc_time_index == False:
                    across_datasets[label_str] = result[:, :equal_end]
                elif acc_time_index == True:
                    across_datasets[label_str] = itr_tr(result[:, :equal_end], time_sl[:equal_end], classification_mode, bin_dt)
            else:
                across_datasets_std[label_str] = np.vstack(
                    (
                        across_datasets_std[label_str],
                        across_subjects_std[:equal_end] / np.sqrt(result.shape[0])
                    )
                )
                if acc_time_index == False:
                    across_datasets[label_str] = np.vstack((across_datasets[label_str], result[:, :equal_end]))
                elif acc_time_index == True:
                    across_datasets[label_str] = np.vstack(
                        (across_datasets[label_str],
                        itr_tr(result[:, :equal_end], time_sl[:equal_end], classification_mode, bin_dt)
                        )
                    )
            
            if d == len(datas) - 1:
                for ax0x in da_axes:
                    ax0x.plot(
                        time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
                        np.mean(across_datasets[label_str], axis=0),
                        c=color,
                        linewidth=linewidths[0],
                        zorder=3,
                    )
                    if len(datas) <= 1:
                        ads = across_datasets_std[label_str]
                    else:
                        ads = np.mean(across_datasets_std[label_str], axis=0)
                    ax0x.fill_between(
                        time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
                        np.mean(across_datasets[label_str], axis=0) - ads,
                        np.mean(across_datasets[label_str], axis=0) + ads,
                        color=color,
                        alpha=0.2,
                        zorder=3,
                    )

        # Sliding window filtering.
        if (
            label_str == "Beta band filter & CSP (15-30 Hz / slw)"
            or label_str == "Mu band filter & CSP (6-15 Hz / slw)"
            or label_str == "Mu-beta band filter & CSP (6-30 Hz / slw)"
        ):
            # Time-resolved decoding accuracy.
            sl_axes[0].plot(
                time_sl if acc_time_index == False else new_time_sl,
                across_subjects,
                c=color,
                linewidth=linewidths[0],
                label=label_str if d == 0 and classification_mode == "sliding" else "",
                zorder=2,
            )

            # Across-subjects std.
            sl_axes[0].fill_between(
                time_sl if acc_time_index == False else new_time_sl,
                across_subjects - across_subjects_std,
                across_subjects + across_subjects_std,
                color=color,
                alpha=0.2,
                zorder=2,
            )

            # Across-datasets accuracy and std.
            if d == 0:
                across_datasets_std[label_str] = across_subjects_std[:equal_end]
                if acc_time_index == False:
                    across_datasets[label_str] = result[:, :equal_end]
                elif acc_time_index == True:
                    across_datasets[label_str] = itr_tr(result[:, :equal_end], time_sl[:equal_end], classification_mode, bin_dt)
            else:
                across_datasets_std[label_str] = np.vstack(
                    (
                        across_datasets_std[label_str],
                        across_subjects_std[:equal_end] / np.sqrt(result.shape[0])
                    )
                )
                if acc_time_index == False:
                    across_datasets[label_str] = np.vstack((across_datasets[label_str], result[:, :equal_end]))
                elif acc_time_index == True:
                    across_datasets[label_str] = np.vstack(
                        (across_datasets[label_str],
                        itr_tr(result[:, :equal_end], time_sl[:equal_end], classification_mode, bin_dt)
                        )
                    )
            
            if d == len(datas) - 1:
                da_axes[0].plot(
                    time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
                    np.mean(across_datasets[label_str], axis=0),
                    c=color,
                    linewidth=linewidths[0],
                    zorder=2,
                )
                if len(datas) <= 1:
                    ads = across_datasets_std[label_str]
                else:
                    ads = np.mean(across_datasets_std[label_str], axis=0)
                da_axes[0].fill_between(
                    time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
                    np.mean(across_datasets[label_str], axis=0) - ads,
                    np.mean(across_datasets[label_str], axis=0) + ads,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

            # Permutation cluster test visualization.
            pipe_ids = np.where(np.array(label_strs) == label_str)[0]
            for pid in pipe_ids:
                dec_lines, dec_col = perm_cl_test_sl[pid]
                sl_axes[0].plot(
                    (
                        time_sl[: time_ids_sl[-1] + 1]
                        if acc_time_index == False
                        else new_time_sl[: time_ids_sl[-1] + 1]
                    ),
                    dec_lines[0][: time_ids_sl[-1] + 1] - perm_offset_1,
                    c=dec_col[0],
                    linewidth=linewidths[2],
                )
                sl_axes[0].plot(
                    (
                        time_sl[: time_ids_sl[-1] + 1]
                        if acc_time_index == False
                        else new_time_sl[: time_ids_sl[-1] + 1]
                    ),
                    dec_lines[1][: time_ids_sl[-1] + 1] - perm_offset_1,
                    c=dec_col[1],
                    linewidth=linewidths[2],
                )

            # Global stats ids.
            if d == len(datas) - 1:
                pipe_ids_all["filter"].append(pipe_ids)

        # Sliding window filter bank.
        if (
            label_str == "Filter bank & CSP (15-30 Hz; 5 bands; 3 Hz step / slw)"
            or label_str == "Filter bank & CSP (6-15 Hz; 3 bands; 3 Hz step / slw)"
            or label_str == "Filter bank & CSP (6-30 Hz; 8 bands; 3 Hz step / slw)"
        ):
            # Time-resolved decoding accuracy.
            sl_axes[1].plot(
                time_sl if acc_time_index == False else new_time_sl,
                across_subjects,
                c=color,
                linewidth=linewidths[0],
                zorder=2,
            )

            # Across-subjects std.
            sl_axes[1].fill_between(
                time_sl if acc_time_index == False else new_time_sl,
                across_subjects - across_subjects_std,
                across_subjects + across_subjects_std,
                color=color,
                alpha=0.2,
                zorder=2,
            )

            # Across-datasets accuracy and std.
            if d == 0:
                across_datasets_std[label_str] = across_subjects_std[:equal_end]
                if acc_time_index == False:
                    across_datasets[label_str] = result[:, :equal_end]
                elif acc_time_index == True:
                    across_datasets[label_str] = itr_tr(result[:, :equal_end], time_sl[:equal_end], classification_mode, bin_dt)
            else:
                across_datasets_std[label_str] = np.vstack(
                    (
                        across_datasets_std[label_str],
                        across_subjects_std[:equal_end] / np.sqrt(result.shape[0])
                    )
                )
                if acc_time_index == False:
                    across_datasets[label_str] = np.vstack((across_datasets[label_str], result[:, :equal_end]))
                elif acc_time_index == True:
                    across_datasets[label_str] = np.vstack(
                        (across_datasets[label_str],
                        itr_tr(result[:, :equal_end], time_sl[:equal_end], classification_mode, bin_dt)
                        )
                    )
            
            if d == len(datas) - 1:
                da_axes[1].plot(
                    time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
                    np.mean(across_datasets[label_str], axis=0),
                    c=color,
                    linewidth=linewidths[0],
                    zorder=2,
                )
                if len(datas) <= 1:
                    ads = across_datasets_std[label_str]
                else:
                    ads = np.mean(across_datasets_std[label_str], axis=0)
                da_axes[1].fill_between(
                    time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
                    np.mean(across_datasets[label_str], axis=0) - ads,
                    np.mean(across_datasets[label_str], axis=0) + ads,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )

            # Permutation cluster test visualization.
            pipe_ids = np.where(np.array(label_strs) == label_str)[0]
            for pid in pipe_ids:
                dec_lines, dec_col = perm_cl_test_sl[pid]
                sl_axes[1].plot(
                    (
                        time_sl[: time_ids_sl[-1] + 1]
                        if acc_time_index == False
                        else new_time_sl[: time_ids_sl[-1] + 1]
                    ),
                    dec_lines[0][: time_ids_sl[-1] + 1] - perm_offset_0,
                    c=dec_col[0],
                    linewidth=linewidths[2],
                )
                sl_axes[1].plot(
                    (
                        time_sl[: time_ids_sl[-1] + 1]
                        if acc_time_index == False
                        else new_time_sl[: time_ids_sl[-1] + 1]
                    ),
                    dec_lines[1][: time_ids_sl[-1] + 1] - perm_offset_0,
                    c=dec_col[1],
                    linewidth=linewidths[2],
                )
            
            # Global stats ids.
            if d == len(datas) - 1:
                pipe_ids_all["filter bank"].append(pipe_ids)


    # ----- #
    # Fig 0 adjustments.

    # Title and ylabel.
    ax00.set_ylabel(
        "Decoding score" if acc_time_index == False else "ITR (bits/s)",
        fontsize=fontsizes[0],
    )
    ax00.set_title(
        title_str,
        fontsize=fontsizes[1],
        loc="left",
        fontweight="bold",
    )

    # Filters and windows legends.
    if d == 0:
        xs = [1.80, -3.50]

        if acc_time_index == False:
            ys = [1.10, 1.15]
        elif acc_time_index == True:
            if ati_zoom == False:
                if classification_mode == "incremental":
                    ys = [1.35, 1.40]
                elif classification_mode == "sliding":
                    ys = [0.45, 0.50]
            elif ati_zoom == True:
                ys = [0.25, 0.30]

        ax00.text(x=xs[0], y=ys[0], s="Single filter", fontsize=fontsizes[1])
        ax01.text(x=xs[0], y=ys[0], s="Filter bank", fontsize=fontsizes[1])

    for axi, ax0x in enumerate(tr_axes + sl_axes):
        if acc_time_index == False:
            # Chance level.
            ax0x.hlines(
                0.5,
                time[0],
                time[-1],
                linestyles="dashed",
                colors="grey",
                linewidth=linewidths[1],
                zorder=1,
            )

        # Trial beginning and end.
        if acc_time_index == False:
            ax0x.vlines(
                (exp_time_periods[1], exp_time_periods[2]),
                0.4,
                1.0,
                linestyles="dotted",
                colors="k",
                linewidth=linewidths[1],
                zorder=1,
            )
        elif acc_time_index == True:
            ax0x.vlines(
                (exp_time_periods[1], exp_time_periods[2]),
                0.0,
                1.35 if ati_zoom == False else 0.25,
                linestyles="dotted",
                colors="k",
                linewidth=linewidths[1],
                zorder=1,
            )

        # Y axis limits and ticks.
        if acc_time_index == False:
            ax0x.set_ylim([0.4, 1.09])
            ax0x.set_yticks(np.arange(0.4, 1.01, 0.1))
        elif acc_time_index == True:
            if ati_zoom == False:
                if classification_mode == "incremental":
                    ax0x.set_ylim([0.0, 1.35])
                    ax0x.set_yticks(np.arange(0.0, 1.21, 0.2))
                elif classification_mode == "sliding":
                    ax0x.set_ylim([0.0, 0.5])
                    ax0x.set_yticks(np.arange(0.0, 0.46, 0.1))
            elif ati_zoom == True:
                ax0x.set_ylim([0.0, 0.30])
                ax0x.set_yticks(np.arange(0.0, 0.26, 0.05))

        if axi == 0 or axi > 1:
            pass
        else:
            ax0x.set_yticklabels([])

        # X axis limits and ticks.
        if (
            classification_mode == "incremental"
            or (classification_mode == "sliding" and sl_time_center == "mid")
        ):
            ax0x.set_xticks(np.arange(0.0, time[-1], 1.0))
        elif classification_mode == "sliding":
            if acc_time_index == False and sl_time_center == "start":
                x_ax_lims = np.arange(-1.0, time[-1], 1.0)
            elif acc_time_index == False and sl_time_center == "end":
                x_ax_lims = np.arange(0.0, np.ceil(exp_time_periods[2] + 0.5), 1.0)
            elif acc_time_index == True:
                x_ax_lims = np.arange(new_time_sl[0], new_time_sl[-1], 1.0)
            ax0x.set_xticks(x_ax_lims)

        # Axes labels.
        if d == len(datas) - 1:
            ax0x.set_xlabel("Time (s)", fontsize=fontsizes[1])

        # Axes tick size.
        ax0x.tick_params(axis="both", labelsize=fontsizes[1])

        # Spines.
        ax0x.spines[["top", "right"]].set_visible(False)


# ----- #
# Across-dataset subplot adjustments.

# Title and ylabel.
da_axes[0].set_ylabel(
    "Decoding score" if acc_time_index == False else "ITR (bits/s)",
    fontsize=fontsizes[0],
)

da_axes[1].set_yticklabels([])

for ax0x in da_axes:

    # Chance level.
    if acc_time_index == False:
        ax0x.hlines(
            0.5,
            time[:equal_end][0] if classification_mode == "incremental" else time_sl[:equal_end][0],
            time[:equal_end][-1] if classification_mode == "incremental" else time_sl[:equal_end][-1],
            linestyles="dashed",
            colors="grey",
            linewidth=linewidths[1],
            zorder=1,
        )
    
    # Trial beginning and end.
    if acc_time_index == False:
        ax0x.vlines(
            exp_time_periods[1],
            0.4,
            1.0,
            linestyles="dotted",
            colors="k",
            linewidth=linewidths[1],
            zorder=1,
        )
    elif acc_time_index == True:
        ax0x.vlines(
            exp_time_periods[1]  + bin_dt,
            0.0  + bin_dt,
            1.35 if ati_zoom == False else 0.25,
            linestyles="dotted",
            colors="k",
            linewidth=linewidths[1],
            zorder=1,
        )
    
    # Y axis limits and ticks.
    if acc_time_index == False:
        ax0x.set_ylim([0.4, 1.09])
        ax0x.set_yticks(np.arange(0.4, 1.01, 0.1))
    elif acc_time_index == True:
        if ati_zoom == False:
            if classification_mode == "incremental":
                ax0x.set_ylim([0.0, 1.35])
                ax0x.set_yticks(np.arange(0.0, 1.21, 0.2))
            elif classification_mode == "sliding":
                ax0x.set_ylim([0.0, 0.5])
                ax0x.set_yticks(np.arange(0.0, 0.46, 0.1))
        elif ati_zoom == True:
            ax0x.set_ylim([0.0, 0.30])
            ax0x.set_yticks(np.arange(0.0, 0.26, 0.05))
        
    
    # X axis limits and ticks.
    if (
        classification_mode == "incremental"
        or (classification_mode == "sliding" and sl_time_center == "mid")
    ):
        ax0x.set_xticks(np.arange(0.0, time[:equal_end + 1][-1], 1.0))
    elif classification_mode == "sliding":
        if acc_time_index == False and sl_time_center == "start":
            x_ax_lims = np.arange(-1.0, time_sl[:equal_end][-1], 1.0)
        elif acc_time_index == False and sl_time_center == "end":
            x_ax_lims = np.arange(0.0, np.ceil(exp_time_periods[2] + 0.5), 1.0)
        elif acc_time_index == True:
            x_ax_lims = np.arange(new_time_sl[:equal_end][0], new_time_sl[:equal_end][-1], 1.0)
        ax0x.set_xticks(x_ax_lims)

    # Axes tick size.
    ax0x.tick_params(axis="both", labelsize=fontsizes[1])

    # Spines.
    ax0x.spines[["top", "right"]].set_visible(False)

# Permutation cluster tests.
if (
    classification_mode == "incremental"
    and perm == True
    and "conv" in res_to_vis
    and len(res_to_vis) >= 2
):
    perm_cl_test_all = vis_permutation_cluster_test(
        perm_data_all,
        res_to_vis,
        colors,
        sub_id=None,
        n_perm=n_perm,
        threshold=threshold,
        correction=correction,
    )

    # Permutation cluster test visualization.
    for pid in pipe_ids_all["filter"]:
        dec_lines, dec_col = perm_cl_test_all[int(pid)]
        da_axes[0].plot(
            time[time_ids[0] : equal_end],
            dec_lines[0][time_ids[0] : equal_end] - perm_offset_0,
            c=dec_col[0],
            linewidth=linewidths[2],
        )
        da_axes[0].plot(
            time[time_ids[0] : equal_end],
            dec_lines[1][time_ids[0] : equal_end] - perm_offset_0,
            c=dec_col[1],
            linewidth=linewidths[2],
        )
    for pid in pipe_ids_all["filter bank"]:
        dec_lines, dec_col = perm_cl_test_all[int(pid)]
        da_axes[1].plot(
            time[time_ids[0] : equal_end],
            dec_lines[0][time_ids[0] : equal_end] - perm_offset_0,
            c=dec_col[0],
            linewidth=linewidths[2],
        )
        da_axes[1].plot(
            time[time_ids[0] : equal_end],
            dec_lines[1][time_ids[0] : equal_end] - perm_offset_0,
            c=dec_col[1],
            linewidth=linewidths[2],
        )

elif (
    classification_mode == "sliding"
    and perm == True
    and "conv_sliding" in res_to_vis
    and len(res_to_vis) >= 2
):
    perm_cl_test_sl_all = vis_permutation_cluster_test(
        perm_data_all,
        res_to_vis,
        colors,
        sub_id=None,
        n_perm=n_perm,
        threshold=threshold,
        correction=correction,
    )

    # Permutation cluster test visualization.
    for pid in pipe_ids_all["filter"]:
        dec_lines, dec_col = perm_cl_test_sl_all[int(pid)]
        da_axes[0].plot(
            time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
            dec_lines[0][:equal_end] - perm_offset_0,
            c=dec_col[0],
            linewidth=linewidths[2],
        )
        da_axes[0].plot(
            time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
            dec_lines[1][:equal_end] - perm_offset_0,
            c=dec_col[1],
            linewidth=linewidths[2],
        )
    for pid in pipe_ids_all["filter bank"]:
        dec_lines, dec_col = perm_cl_test_sl_all[int(pid)]
        da_axes[1].plot(
            time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
            dec_lines[0][:equal_end] - perm_offset_0,
            c=dec_col[0],
            linewidth=linewidths[2],
        )
        da_axes[1].plot(
            time_sl[:equal_end] if acc_time_index == False else new_time_sl[:equal_end],
            dec_lines[1][:equal_end] - perm_offset_0,
            c=dec_col[1],
            linewidth=linewidths[2],
        )


# ----- #
# Legends.
leg_cols = 1
fig0.legend(
    frameon=False,
    title="Feature extraction pipeline",
    alignment="left",
    fontsize=fontsizes[1],
    title_fontsize=fontsizes[0],
    ncols=leg_cols,
    loc="upper center",
)


# ----- #
# Optional saving.
if savefigs == True:
    fig0_name = savepath + "dataset_average_time_resolved{}_decoding_{}{}{}.{}".format(
        clm_str,
        "scores" if acc_time_index == False else "index",
        "_zoom" if acc_time_index == True and ati_zoom == True else "",
        fooof_save_str,
        plot_format,
    )
    fig0.savefig(fig0_name, dpi=dpi, facecolor="w", edgecolor="w")
else:
    plt.show()
