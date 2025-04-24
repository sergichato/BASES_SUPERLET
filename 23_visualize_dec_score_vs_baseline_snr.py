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
# Hyperparameters.
metric = "rocauc"
if metric == "rocauc":
    metric_str = "score"
elif metric == "accuracy":
    metric_str = metric

remove_fooof = False
if remove_fooof == True:
    fooof_save_str = ""
elif remove_fooof == False:
    fooof_save_str = "_nfs"

clm_str_tr = "_tr"
clm_str_sl = "_sl"

savefigs = False  # True, False

perm = True  # True, False

plot_format = "png"  # "pdf", "png"


# ----- #
# Dataset selection.
datas = [
    "zhou2016",
    "2014004",
    "2014001",
    "weibo2014",
    "munichmi",
    "cho2017",
    # "dreyer2023",
]

# Mode.
mode = "local"  # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/"


# ----- #
# Results selestion.
classification_mode = "both"   # "incremental", "both"
if classification_mode == "incremental":
    res_to_vis = [
        "conv",
        "fb_beta",
        # "fb_mu",
        "fb_mubeta",
        "fb_1_beta",
        "fb_1_mubeta",
        "conv_riemann",
    ]
elif classification_mode == "both":
    # HARD CODED ASSUMPTION: sliding window results
    # always come last in the list, and convolution is
    # always the first of either category.
    res_to_vis = [
        "conv",
        "fb_beta",
        # "fb_mu",
        "fb_mubeta",
        "fb_1_beta",
        # "fb_1_mu",
        "fb_1_mubeta",
        "conv_sliding",
        "fb_beta_sliding",
        # "fb_mu_sliding",
        "fb_mubeta_sliding",
        "fb_1_beta_sliding",
        # "fb_1_mu_sliding",
        "fb_1_mubeta_sliding",
    ]

    sl_res_id = int(np.where(np.array(res_to_vis)=="conv_sliding")[0])
    fb_1_res_id = int(np.where(np.array(res_to_vis)=="fb_1_mubeta")[0])

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
    fig2 = plt.figure(
        constrained_layout=False,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
    gs2 = fig2.add_gridspec(
        nrows=1,
        ncols=2,
        wspace=0.50,
        left=0.15,
        right=0.80,
        top=0.75,
        bottom=0.10,
    )

else:
    fig2 = plt.figure(constrained_layout=False, figsize=(7.0, 4.0), dpi=dpi)
    gs2 = fig2.add_gridspec(
        nrows=1,
        ncols=2,
        wspace=0.40,
        left=0.07,
        right=0.95,
        top=0.80,
        bottom=0.10,
    )


# ----- #
# Data loading.
snrs_all = []
for d, data in enumerate(datas):
    if data == "zhou2016":
        variables_path = "{}zhou_2016/variables.json".format(basepath)
        title_str = "Zhou 2016"
        data_color = "cornflowerblue"
        c3c4_ids = [3, 5]
    elif data == "2014004":
        variables_path = "{}2014_004/variables.json".format(basepath)
        title_str = "BNCI 2014-004"
        data_color = "orchid"
        c3c4_ids = [0, 2]
    elif data == "2014001":
        variables_path = "{}2014_001/variables.json".format(basepath)
        title_str = "BNCI 2014-001"
        data_color = "darkgoldenrod"
        c3c4_ids = [3, 5]
    elif data == "munichmi":
        variables_path = "{}munichmi/variables.json".format(basepath)
        title_str = "Munich MI\n(Grosse-Wentrup 2009)"
        data_color = "orangered"
        c3c4_ids = [4, 8]
    elif data == "cho2017":
        variables_path = "{}cho_2017/variables.json".format(basepath)
        title_str = "Cho 2017"
        data_color = "yellow"
        c3c4_ids = [3, 5]
    elif data == "weibo2014":
        variables_path = "{}weibo_2014/variables.json".format(basepath)
        title_str = "Weibo 2014"
        data_color = "mediumseagreen"
        c3c4_ids = [3, 5]
    elif data == "dreyer2023":
        variables_path = "{}dreyer_2023/variables.json".format(basepath)
        title_str = "Dreyer 2023"
        data_color = "navy"
        c3c4_ids = [3, 5]

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

    # Time in 100ms non-overlapping windows.
    time = np.around(
        np.arange(
            exp_time_periods[1] - 0.5, exp_time_periods[2] + 0.5 + bin_dt, bin_dt
        ),
        2,
    )

    # Time in 1000ms sliding (by 50ms) windows.
    time_sl = np.around(
        np.arange(
            exp_time_periods[1] - 0.5, exp_time_periods[2] + 0.5 + bin_dt / 2, bin_dt / 2
        ),
        3,
    )

    tmin = experimental_vars["tmin"]
    tmax = experimental_vars["tmax"]

    time_points = [0.0, 0.5, 1.0, 2.0, exp_time_periods[2]]
    time_ids = [np.where(time == tp)[0][0] for tp in time_points]
    time_ids_sl = [np.where(time_sl == tp)[0][0] for tp in time_points]

    # Time in experiment.
    sfreq = experimental_vars["sfreq"]
    exp_time = np.linspace(tmin, tmax, int((np.abs(tmax - tmin)) * sfreq) + 1)
    exp_time = np.around(exp_time, decimals=3)
    task_time_lims = [exp_time_periods[1], exp_time_periods[2]]
    base_start = -0.5

    try:
        base_start_id = np.where(exp_time == base_start)[0][0]
    except:
        base_start_id = np.where(exp_time >= base_start)[0][0][0]
    try:
        trial_start_id = np.where(exp_time == task_time_lims[0])[0][0]
    except:
        trial_start_id = np.where(exp_time >= task_time_lims[0])[0][0][0]
    try:
        trial_end_id = np.where(exp_time == task_time_lims[1])[0][0]
    except:
        trial_end_id = np.where(exp_time <= task_time_lims[1])[0][0][-1]

    # ----- #
    # Data loading and computation of baseline power.
    snrs = []
    for sub in subs:
        # Suject-specific directory.
        sub_dir = join(savepath, "sub_{}/".format(sub))

        print("Loading and processing S{} TF and FOOOF data...".format(sub))

        # Generic mu frequency range.
        freq_step = 0.5
        freqs = np.arange(1.0, 43.25, freq_step)

        upto_mu_range = np.where(
            np.logical_and(freqs >= power_band[0], freqs <= power_band[1])
        )[0]

        # Channel-specific mu band indexes.
        with open(
            join(
                sub_dir, "mu_search_ranges_superlets.pkl"
            ),
            "rb",
        ) as pickle_file:
            msr = pickle.load(pickle_file)
        
        msr_c3 = msr[c3c4_ids[0]]
        
        if np.isnan(msr_c3).all():
            msr_c3 = upto_mu_range
        
        msr_c4 = msr[c3c4_ids[1]]
        if np.isnan(msr_c4).all():
            msr_c4 = upto_mu_range

        # Channel-specific mu band FOOOF fits.
        with open(
            join(
                sub_dir, "mu_fooof_thresholds_superlets.pkl"
            ),
            "rb",
        ) as pickle_file:
            aps = pickle.load(pickle_file)
        
        aps_c3 = aps[c3c4_ids[0]]       
        aps_c4 = aps[c3c4_ids[1]]

        # Subject's TF data loading.
        tfs = np.load(join(sub_dir, "tfs_superlets.npy"))
        av_psd_c3 = np.mean(tfs[:, c3c4_ids[0], msr_c3, base_start_id:trial_start_id], axis=(0, -1))
        av_psd_c4 = np.mean(tfs[:, c3c4_ids[0], msr_c4, base_start_id:trial_start_id], axis=(0, -1))

        del tfs

        # Signal-to-noise ratio.
        snr = np.mean(
            (
            np.max( 10 * np.log10( av_psd_c3 / aps_c3 ) ) if len(aps_c3) != 0 else 0,
            np.max( 10 * np.log10( av_psd_c4 / aps_c4 ) ) if len(aps_c4) != 0 else 0,
            ),
        )
        snrs.append(snr)
    snrs_all.append(snrs)

    # ----- #
    # Loading of decoding results.
    # Dataset results.
    results = []
    results_std = []
    perm_data = []
    label_strs = []
    colors = []

    # All convolution results.
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

    # All Riemannian results.
    if "conv_riemann" in res_to_vis:
        label_strs.append("Convolution & TGSP")
        colors.append("royalblue")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_beta_band_riemann{}_conv_waves{}.npy".format(
                        metric, fooof_save_str, clm_str_tr
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_beta_band_riemann{}_conv_waves{}.npy".format(
                    metric, fooof_save_str, clm_str_tr
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_beta_band_riemann{}_conv_waves{}.npy".format(
                        metric, fooof_save_str, clm_str_tr
                    )
                ),
                axis=-1,
            ),
        )
    
    if "fb_beta_riemann" in res_to_vis:
        label_strs.append(
            "Filter bank & MDM ({}-{} Hz; {} bands; {} Hz step)".format(
                filter_banks[1][0][0],
                filter_banks[1][-1][1],
                len(filter_banks[1]),
                filter_banks[1][0][1] - filter_banks[1][0][0],
            ),
        )
        colors.append("limegreen")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_riemann_{}_{}{}.npy".format(
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
                + "std_{}_power_filter_bank_{}_riemann_{}_{}{}.npy".format(
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
                    + "mean_{}_power_filter_bank_{}_riemann_{}_{}{}.npy".format(
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

    if "fb_mubeta_riemann" in res_to_vis:
        label_strs.append(
            "Filter bank & MDM ({}-{} Hz; {} bands; {} Hz step)".format(
                filter_banks[2][0][0],
                filter_banks[2][-1][1],
                len(filter_banks[2]),
                filter_banks[2][0][1] - filter_banks[2][0][0],
            ),
        )
        colors.append("teal")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_{}_riemann_{}_{}{}.npy".format(
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
                + "std_{}_power_filter_bank_{}_riemann_{}_{}{}.npy".format(
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
                    + "mean_{}_power_filter_bank_{}_riemann_{}_{}{}.npy".format(
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
    
    if "fb_1_beta_riemann" in res_to_vis:
        label_strs.append(
            "Filter bank & MDM ({}-{} Hz; {} band)".format(
                filter_banks[3][0][0],
                filter_banks[3][-1][1],
                len(filter_banks[3]),
            ),
        )
        colors.append("limegreen")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_1_riemann_{}_{}{}.npy".format(
                        metric, filter_banks[4][0][0], filter_banks[4][-1][1], clm_str_tr
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_1_riemann_{}_{}{}.npy".format(
                    metric, filter_banks[4][0][0], filter_banks[4][-1][1], clm_str_tr
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_1_riemann_{}_{}{}.npy".format(
                        metric, filter_banks[4][0][0], filter_banks[4][-1][1], clm_str_tr
                    )
                ),
                axis=-1,
            ),
        )

    if "fb_1_mubeta_riemann" in res_to_vis:
        label_strs.append(
            "Filter bank & MDM ({}-{} Hz; {} band)".format(
                filter_banks[5][0][0],
                filter_banks[5][-1][1],
                len(filter_banks[5]),
            ),
        )
        colors.append("teal")
        results.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_1_riemann_{}_{}{}.npy".format(
                        metric, filter_banks[5][0][0], filter_banks[5][-1][1], clm_str_tr
                    )
                ),
                axis=(2, 3),
            ),
        )
        results_std.append(
            np.load(
                savepath
                + "std_{}_power_filter_bank_riemann_1_{}_{}{}.npy".format(
                    metric, filter_banks[5][0][0], filter_banks[5][-1][1], clm_str_tr
                )
            ),
        )
        perm_data.append(
            np.mean(
                np.load(
                    savepath
                    + "mean_{}_power_filter_bank_1_riemann_{}_{}{}.npy".format(
                        metric, filter_banks[5][0][0], filter_banks[5][-1][1], clm_str_tr
                    )
                ),
                axis=-1,
            ),
        )
    
    # All sliding window results.
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


    # ----- #
    # Plot.

    # SNR in mu band and decoding score.
    if d == 0:
        gs20 = gs2[0].subgridspec(1, 1)
        ax20 = fig2.add_subplot(gs20[0])
        gs21 = gs2[1].subgridspec(1, 1)
        ax21 = fig2.add_subplot(gs21[0])


    # ----- #
    # Fig 2.
    Y1 = np.mean(results[0][:, time_ids[-1] - 1 : time_ids[-1] + 2], axis=-1)
    Y2 = np.mean(results[fb_1_res_id][:, time_ids[-1] - 1 : time_ids[-1] + 2], axis=-1)
    ax20.scatter(
        snrs,
        Y1,
        s=3.0,
        c=data_color,
        label=title_str,
    )

    ax21.scatter(
        snrs,
        Y2,
        s=3.0,
        c=data_color,
    )

    # X axis limits.
    if d == len(datas) - 1:
        snrs_all = np.hstack(snrs_all)
        snr_lims = np.max(snrs_all)
        ax20.set_xlim([-0.5, snr_lims + 0.5])
        ax21.set_xlim([-0.5, snr_lims + 0.5])

    # Axes labels and titles.
    ax20.set_xlabel("Baseline SNR (dB)", fontsize=fontsizes[1])
    ax20.set_ylabel("Decoding score", fontsize=fontsizes[1])
    ax20.set_title(label_strs[0], fontsize=fontsizes[1])

    ax21.set_xlabel("Baseline SNR (dB)", fontsize=fontsizes[1])
    ax21.set_title(label_strs[4], fontsize=fontsizes[1])

    # Y axis limits and ticks.
    ax20.set_ylim([0.4, 1.0])
    ax20.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax20.xaxis.offsetText.set_fontsize(fontsizes[1])

    ax21.set_ylim([0.4, 1.0])
    ax21.set_yticklabels([])
    ax21.xaxis.offsetText.set_fontsize(fontsizes[1])

    # Axes tick size.
    ax20.tick_params(axis="both", labelsize=fontsizes[1])
    ax21.tick_params(axis="both", labelsize=fontsizes[1])

    # Spines.
    ax20.spines[["top", "right"]].set_visible(False)
    ax21.spines[["top", "right"]].set_visible(False)

    # Grid.
    ax20.grid(visible=False)
    ax21.grid(visible=False)

    print("\n")

    # Correlation.
    cor1 = np.corrcoef(snrs, Y1)[0,1]
    m1, b1 = np.polyfit(snrs, Y1, 1)
    ax20.plot(
        snrs,
        m1 * np.array(snrs) + b1,
        c=data_color,
    )
    ax20.text(
        np.max(snrs),
        np.max(m1 * np.array(snrs) + b1),
        "{}".format(np.around(cor1, 3)),
        color=data_color,
        fontsize=fontsizes[1],
    )

    cor2 = np.corrcoef(snrs, Y2)[0,1]
    m2, b2 = np.polyfit(snrs, Y2, 1)
    ax21.plot(
        snrs,
        m2 * np.array(snrs) + b2,
        c=data_color,
    )
    ax21.text(
        np.max(snrs),
        np.max(m2 * np.array(snrs) + b2),
        "{}".format(np.around(cor2, 3)),
        color=data_color,
        fontsize=fontsizes[1],
    )


# ----- #
# Legends.
fig2.legend(
    frameon=False,
    title="Dataset",
    alignment="left",
    fontsize=fontsizes[1],
    title_fontsize=fontsizes[0],
    ncols=2,
)


# ----- #
# Optional saving.
if savefigs == True:
    fig2_name = savepath + "dataset_average_snr_to_decoding_scores{}.{}".format(
        fooof_save_str, plot_format
    )
    fig2.savefig(fig2_name, dpi=dpi, facecolor="w", edgecolor="w")
else:
    plt.show()
