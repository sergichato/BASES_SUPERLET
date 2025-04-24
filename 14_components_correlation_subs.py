import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from burst_space import BurstSpace
from help_funcs import load_exp_variables


# ----- #
# Hyperparameters.
metric = "rocauc"
if metric == "rocauc":
    metric_str = "score"
elif metric == "accuracy":
    metric_str = metric

band = "beta"  # "beta", "mu"
if band == "beta":
    band_letter = "β"
elif band == "mu":
    band_letter = "μ"

limit_hspace = True
if limit_hspace == True:
    hspace_str = "_sel"
elif limit_hspace == False:
    hspace_str = ""

remove_fooof = True
if remove_fooof == True:
    fooof_save_str = ""
elif remove_fooof == False:
    fooof_save_str = "_nfs"

threshold = False  # 0.65
if threshold != False:
    threshold_str = "_thr"
elif threshold == False:
    threshold_str = ""

savefigs = True

plot_format = "pdf"  # "pdf", "png"


# ----- #
# Additional variables.
trials_fraction = 1.0
solver = "pca"
if solver == "pca":
    nc = 0.99
elif solver == "csp":
    nc = 8

n_comps_to_vis = 8


# ----- #
# Datasets.
datas = ["zhou2016", "2014004", "2014001", "munichmi", "weibo2014", "cho2017"]

# Mode.
mode = "cluster"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

titles = [
    "Zhou 2016",
    "BNCI 2014-004",
    "BNCI 2014-001",
    "MunichMI",
    "Weibo 2014",
    "Cho 2017",
]

variables_paths = [
    "{}zhou_2016/variables.json".format(basepath),
    "{}2014_004/variables.json".format(basepath),
    "{}2014_001/variables.json".format(basepath),
    "{}munichmi/variables.json".format(basepath),
    "{}weibo_2014/variables.json".format(basepath),
    "{}cho_2017/variables.json".format(basepath),
]

# Channels of interest.
channel_ids = [[3, 5], [0, 2], [3, 5], [4, 8], [3, 5], [3, 5]]

# PCA components.
pca_comps = []

for data, variables_path, cids in zip(datas, variables_paths, channel_ids):
    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    savepath = experimental_vars["dataset_path"]

    subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
    if data == "cho2017":
        # Some subjects are not included in the dataset.
        subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

    # Classification results.
    burst_accs = np.load(
        savepath
        + "mean_{}_stratified_{}_bursts_pca{}{}.npy".format(
            metric, band, fooof_save_str, hspace_str
        )
    )

    mean_burst_accs = np.mean(burst_accs, axis=1)

    best_sub = np.argmax(mean_burst_accs)
    worst_sub = np.argmin(mean_burst_accs)

    ind_dicts = []

    # ----- #
    # Burst dictionary creation and application of dimensionality reduction.
    # Only look into channels C3 and C4.
    # Use 'band="mu"' in order to load mu band bursts, instead of beta bursts.
    for s in [best_sub, worst_sub]:
        bspace = BurstSpace(
            experimental_vars,
            subjects[s],
            trials_fraction=trials_fraction,
            channel_ids=cids,
            band="beta",
            remove_fooof=remove_fooof,
        )

        # Fit all available trials in order to produce the visualization.
        # Parameter 'purpose' has to be set to "plotting".
        bspace.fit_transform(solver=solver, n_components=nc, purpose="plotting")

        ind_dicts.append(bspace.drm.components_)

    pca_comps.append(ind_dicts)

# ----- #
# Correlations of components across datasets.
correlation_matrices = []

for data_comps in pca_comps:
    absolute_cor = np.zeros((n_comps_to_vis, n_comps_to_vis))

    for i in range(n_comps_to_vis):
        orig_comp = data_comps[0][i, :]

        for j in range(n_comps_to_vis):
            rest_comp = data_comps[1][j, :]

            # Absolute correlation.
            absolute_cor[i, j] = np.abs(np.corrcoef(orig_comp, rest_comp)[0, 1])

            if threshold != False and absolute_cor[i, j] < threshold:
                absolute_cor[i, j] = 0.0

    correlation_matrices.append(absolute_cor)


# ----- #
# Figure initialization.
screen_res = [1920, 972]
dpi = 300
title_size = 8
label_size = 8
tick_size = 4

if savefigs == True:
    fig = plt.figure(
        constrained_layout=False,
        figsize=(7, 9),
        dpi=dpi,
    )
elif savefigs == False:
    fig = plt.figure(
        constrained_layout=False,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    wspace=0.1,
    left=0.05,
    right=0.93,
    top=0.95,
    bottom=0.05,
    width_ratios=[0.99, 0.01],
)

n_rows = int(len(datas) / 2)
n_cols = 2

gs_0 = gs[0].subgridspec(nrows=n_rows, ncols=n_cols, hspace=0.2, wspace=0.05)
gs_1 = gs[1].subgridspec(nrows=n_rows, ncols=1)

for c, corr_matrix in enumerate(correlation_matrices):
    # Subplots.
    rid = c // 2
    cid = c % 2

    ax = fig.add_subplot(gs_0[rid, cid])

    im = ax.imshow(
        corr_matrix, cmap="coolwarm", aspect="equal", vmin=0, vmax=1, origin="lower"
    )

    # Ticks and labels.
    ax.set_xticks(np.arange(0, n_comps_to_vis, 1), fontweight=tick_size)
    ax.set_yticks(np.arange(0, n_comps_to_vis, 1), fontweight=tick_size)
    ax.set_xticks(np.arange(-0.5, n_comps_to_vis, 1), minor=True, fontweight=tick_size)
    ax.set_yticks(np.arange(-0.5, n_comps_to_vis, 1), minor=True, fontweight=tick_size)

    if cid == 0:
        ax.set_yticklabels(np.arange(1, n_comps_to_vis + 1, 1), fontsize=label_size)
    else:
        ax.set_yticklabels([])

    if rid == n_rows - 1:
        ax.set_xticklabels(np.arange(1, n_comps_to_vis + 1, 1), fontsize=label_size)
    else:
        ax.set_xticklabels([])

    ax.tick_params(which="minor", bottom=False, left=False)

    if cid == 0:
        ax.set_ylabel("Best subject components", fontsize=label_size)

    if rid == n_rows - 1:
        ax.set_xlabel("Worst subject components", fontsize=label_size)

    # Titles.
    ax.set_title(titles[c], fontsize=title_size, fontweight="bold")

    # Colorbars.
    if cid == 0:
        c_ax = fig.add_subplot(gs_1[rid])
        cb = fig.colorbar(im, cax=c_ax)
        cb.set_label(label="Abs(corr.)", fontsize=title_size, fontweight="bold")
        cb.ax.tick_params(labelsize=label_size)

# Optional saving.
if savefigs == True:
    fig.savefig(
        "/home/sotpapad/Codes/cho_2017/comps_cors_subs{}{}.pdf".format(
            fooof_save_str, threshold_str
        ),
        dpi=dpi,
        facecolor="w",
        edgecolor="w",
    )
elif savefigs == True:
    plt.show()
