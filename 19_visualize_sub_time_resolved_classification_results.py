import numpy as np
import matplotlib.pyplot as plt

from help_funcs import load_exp_variables, vis_permutation_cluster_test


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

classification_mode = "incremental"
if classification_mode == "incremental":
    clm_str = clm_str_tr
elif classification_mode == "sliding":
    clm_str = clm_str_sl

sl_time_center = "start"  # "start", "mid", "end"

savefigs = True         # True, False

perm = True             # True, False

plot_format = "png"     # "pdf", "png"
dpi = 300
screen_res = [1920, 972]


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
    basepath = "/crnldata/cophy/Jeremie/Sotiris/bebop/"


# ----- #
# Results selection.
if classification_mode == "incremental":
    #"""
    res_to_vis = [
        "conv",
        "fb_beta",
        "fb_mubeta",
        "fb_mu"
    ]
    #"""
    """
    res_to_vis = [
        "conv",
        "fb_1_beta",
        "fb_1_mubeta",
        "fb_1_mu"
    ]
    """
elif classification_mode == "sliding":
    #"""
    res_to_vis = [
        "conv_sliding",
        "fb_beta_sliding",
        "fb_mubeta_sliding",
        "fb_mu_sliding"
    ]
    #"""
    """
    res_to_vis = [
        "conv_sliding",
        "fb_1_beta_sliding",
        "fb_1_mubeta_sliding",
        "fb_1_mu_sliding"
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

for data in datas:
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
    elif data == "physionet":
        variables_path = "{}physionet/variables.json".format(basepath)
        title_str = "PhysionetMI"
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

    if classification_mode == "incremental":
        time = np.around(
            np.arange(
                exp_time_periods[1] - 0.5, exp_time_periods[2] + 0.5 + bin_dt, bin_dt
            ),
            2,
        )

    elif classification_mode == "sliding":
        
        if sl_time_center == "mid":

            p_lim = 0.5
            m_lim = 0.5
            if data == "cho2017":
                p_lim += 0.05
        
        elif sl_time_center == "start":

            p_lim = 0.0
            m_lim = 1.0
            if data == "cho2017":
                p_lim += 0.05
            
        elif sl_time_center == "end":

            p_lim = 1.0
            m_lim = 0.0
            if data == "cho2017":
                p_lim += 0.05

        time = np.around(
            np.arange(
                exp_time_periods[1] - m_lim, exp_time_periods[2] + p_lim + bin_dt / 2, bin_dt / 2
            ),
            3,
        )
    
    time_points = [0.0, 0.5, 1.0, 2.0, exp_time_periods[2]]
    time_ids = [np.where(time == tp)[0][0] for tp in time_points]

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


    # ------ #
    # Figure.

    linewidths = [1.5, 0.75, 1.0]
    fontsizes = [4, 6]

    if savefigs == False:
        fig = plt.figure(
            constrained_layout=False,
            figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
            dpi=dpi,
        )
    elif savefigs == True:
        fig = plt.figure(constrained_layout=False, figsize=(7, 4), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=1,
        wspace=0.05,
        left=0.08,
        right=0.97,
        top=0.82,
        bottom=0.10,
    )

    # Choose "best" subjects w.r.t. convolution, for datasets consisting
    # of more than 10 subjects.
    if data == "cho2017" or data == "dreyer2023":
        # Choose best subjects with respect to burst waveforms
        # classification scores.
        trial_results = np.load(
            savepath
            + "mean_{}_power_beta_band_csp{}_conv_waves{}.npy".format(
                metric, fooof_save_str, clm_str,
            )
        )
        order = np.argsort(np.nanmean(trial_results, axis=(1, 2, 3)))
        subs_orig = np.copy(subs)
        subs = np.array(subs)[order[-10:]]
    
    if data == "zhou2016":
        cols = 2
    else:
        cols = 5
    rows = 2
    split = int(np.ceil(len(subs) / 2))

    gs00 = gs[0].subgridspec(rows, cols, wspace=0.2, hspace=0.5)

    # Subplots.
    for s in range(len(subs)):

        row = s // split
        col = s % split

        ax = fig.add_subplot(gs00[row, col])

        sub_res = []
        sub_stds = []

        # Retrieve correct time-resolved results and corresponding std.
        for i, (res, res_std) in enumerate(zip(results, results_std)):
            if data == "cho2017" or data == "dreyer2023":
                s_orig = int(np.where(subs_orig == subs[s])[0])
                sub_res.append(res[s_orig, :])
                sub_stds.append(res_std[s_orig, :])
            else:
                sub_res.append(res[s, :])
                sub_stds.append(res_std[s, :])

        # Plot.
        for i, (res, res_std, lab, color) in enumerate(
            zip(sub_res, sub_stds, label_strs, colors)
        ):
            if s == 0:
                ax.plot(time, res, c=color, linewidth=linewidths[0], label=lab)
            else:
                ax.plot(time, res, c=color, linewidth=linewidths[0])
            ax.fill_between(time, res - res_std, res + res_std, color=color, alpha=0.2)

        # Chance level.
        ax.hlines(
            0.5,
            time[0],
            time[-1],
            linestyles="dashed",
            colors="grey",
            linewidth=linewidths[1],
        )

        # Trial beginning and end.
        ax.vlines(
            (exp_time_periods[1], exp_time_periods[2]),
            0.4,
            1.0,
            linestyles="dotted",
            colors="k",
            linewidth=linewidths[1],
        )

        # Y axis limits and ticks.
        ax.set_ylim([0.4, 1.02])
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        if col != 0:
            ax.set_yticklabels([])

        # Axes labels.
        if row == 1:
            ax.set_xlabel("Time (s)", fontsize=fontsizes[0])
        if col == 0:
            ax.set_ylabel("Decoding score", fontsize=fontsizes[0])

        # Axes tick size.
        ax.tick_params(axis="both", labelsize=fontsizes[0])

        # Subplot title.
        if data == "cho2017" or data == "physionet":
            ax.set_title(
                "S{}".format(subs[s]), fontsize=fontsizes[1], fontweight="bold"
            )
        else:
            ax.set_title("S{}".format(s + 1), fontsize=fontsizes[1], fontweight="bold")

        # Spines.
        ax.spines[["top", "right"]].set_visible(False)

        # Visualize permutation cluster tests.
        if (
            ("conv" in res_to_vis
            or "conv_sliding" in res_to_vis)
            and len(res_to_vis) >= 2
        ):
            perm_cl_test = vis_permutation_cluster_test(
                perm_data,
                res_to_vis,
                colors,
                s,
            )

            for dec_lines in perm_cl_test:
                ax.plot(
                    time[time_ids[0] : time_ids[-1] + 1],
                    dec_lines[0][0][time_ids[0] : time_ids[-1] + 1],
                    c=dec_lines[1][0],
                    linewidth=linewidths[2],
                )
                ax.plot(
                    time[time_ids[0] : time_ids[-1] + 1],
                    dec_lines[0][1][time_ids[0] : time_ids[-1] + 1],
                    c=dec_lines[1][1],
                    linewidth=linewidths[2],
                )

            # Adjust upper y limit based on number of statistical comparisons,
            ax.set_ylim([0.4, 1.00 + 0.03 * len(perm_cl_test)])

    # Figure title.
    fig.suptitle(title_str, fontsize=fontsizes[1], fontweight="bold")

    # Legend.
    leg_cols = len(res_to_vis) // 2 + len(res_to_vis) % 2
    fig.legend(
        frameon=False,
        title="Feature extraction pipeline",
        alignment="left",
        fontsize=fontsizes[0],
        title_fontsize=fontsizes[1],
        ncols=leg_cols,
    )

    # Optional saving.
    if savefigs == True:
        fig_name = savepath + "time_resolved_decoding{}{}_vn{}.{}".format(
            clm_str, fooof_save_str, len(res_to_vis), plot_format
        )
        fig.savefig(fig_name, dpi=dpi, facecolor="w", edgecolor="w")
    elif savefigs == False:
        plt.show()
