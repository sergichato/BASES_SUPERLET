import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.signal import resample_poly
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

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

remove_fooof = False
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
mode = "local"    # "local", "cluster"
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

# Sampling frequencies.
sfreqs = []

for data, variables_path, cids in zip(datas, variables_paths, channel_ids):
    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    sfreqs.append(experimental_vars["sfreq"])

    subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
    if data == "cho2017":
        # Some subjects are not included in the dataset.
        subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

    # ----- #
    # Burst dictionary creation and application of dimensionality reduction.
    # Only look into channels C3 and C4.
    # Use 'band="mu"' in order to load mu band bursts, instead of beta bursts.
    bspace = BurstSpace(
        experimental_vars,
        subjects,
        trials_fraction=trials_fraction,
        channel_ids=cids,
        band="beta",
        remove_fooof=remove_fooof,
    )

    # Fit all available trials in order to produce the visualization.
    # Parameter 'purpose' has to be set to "plotting".
    bspace.fit_transform(solver=solver, n_components=nc, purpose="plotting")

    pca_comps.append(bspace.drm.components_)

# ----- #
# Correlations of components across datasets.
correlation_matrices = []

for d, (data_pca_comps, sfreq_orig) in enumerate(zip(pca_comps, sfreqs)):
    # Correlation to all other datasets.
    pca_comps_rest = np.delete(np.copy(pca_comps), d)

    sfreqs_rest = np.delete(np.copy(sfreqs), d)

    data_correlation_matrices = []

    for comps_rest, sfreq_rest in zip(pca_comps_rest, sfreqs_rest):
        absolute_cor = np.zeros((n_comps_to_vis, n_comps_to_vis))

        for i in range(n_comps_to_vis):
            for j in range(n_comps_to_vis):
                orig_comp = data_pca_comps[i, :]
                rest_comp = comps_rest[j, :]

                # Dynamic time warping of components when comparing datasets
                # with different sampling rates.
                if sfreq_orig != sfreq_rest:
                    # ~Double sampling rate can se solved with downsampling.
                    if sfreq_orig <= sfreq_rest / 2:
                        rest_comp = resample_poly(rest_comp, 1, 2)[: len(orig_comp)]

                    elif sfreq_orig >= sfreq_rest * 2:
                        orig_comp = resample_poly(orig_comp, 1, 2)[: len(rest_comp)]

                    # Almost the same sampling rate can use a "hack", because
                    # the number of samples only differs by 2.
                    elif (sfreq_orig <= sfreq_rest) and (sfreq_orig > sfreq_rest / 2):
                        rest_comp = rest_comp[: len(orig_comp)]

                    elif (sfreq_rest <= sfreq_orig) and (sfreq_rest > sfreq_orig / 2):
                        orig_comp = orig_comp[: len(rest_comp)]

                    # Weird sampling relationships need dynamic time warping.
                    else:
                        _, path = fastdtw(orig_comp, rest_comp, dist=euclidean)

                        trans_comp_1 = np.zeros((len(path)))
                        trans_comp_2 = np.zeros((len(path)))

                        for k, p in enumerate(path):
                            trans_comp_1[k] = orig_comp[p[0]]
                            trans_comp_2[k] = rest_comp[p[1]]

                        orig_comp = trans_comp_1
                        rest_comp = trans_comp_2

                # Absolute correlation.
                absolute_cor[i, j] = np.abs(np.corrcoef(orig_comp, rest_comp)[0, 1])

                if threshold != False and absolute_cor[i, j] < threshold:
                    absolute_cor[i, j] = 0.0

        data_correlation_matrices.append(absolute_cor)

    correlation_matrices.append(data_correlation_matrices)


# ----- #
# Figure initialization.
screen_res = [1920, 972]
dpi = 300
title_size = 5
label_size = 4
tick_size = 4
hspace = 0.15

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

gs_0 = gs[0].subgridspec(nrows=len(datas), ncols=len(datas) - 1, hspace=hspace)
gs_1 = gs[1].subgridspec(nrows=len(datas), ncols=1, hspace=hspace * 2)

for r, row in enumerate(correlation_matrices):
    ref_data = titles[r]
    test_datas = np.delete(np.copy(titles), r)

    for c, col in enumerate(row):
        # Subplots.
        ax = fig.add_subplot(gs_0[r, c])

        im = ax.imshow(
            col, cmap="coolwarm", aspect="equal", vmin=0, vmax=1, origin="lower"
        )

        # Ticks and labels.
        ax.set_xticks(np.arange(0, n_comps_to_vis, 1), fontweight=tick_size)
        ax.set_yticks(np.arange(0, n_comps_to_vis, 1), fontweight=tick_size)
        ax.set_xticks(
            np.arange(-0.5, n_comps_to_vis, 1), minor=True, fontweight=tick_size
        )
        ax.set_yticks(
            np.arange(-0.5, n_comps_to_vis, 1), minor=True, fontweight=tick_size
        )

        if c == 0:
            ax.set_yticklabels(np.arange(1, n_comps_to_vis + 1, 1), fontsize=label_size)
        else:
            ax.set_yticklabels([])

        if c == len(datas) - 1:
            ax.set_xticklabels(np.arange(1, n_comps_to_vis + 1, 1), fontsize=label_size)
        else:
            ax.set_xticklabels([])

        ax.tick_params(which="minor", bottom=False, left=False)

        if c == 0:
            ax.set_ylabel("Reference dataset components", fontsize=label_size)

        if r == len(datas) - 1:
            ax.set_xlabel("Compared dataset components", fontsize=label_size)

        # Titles.
        if c == 0:
            ax.set_title(
                ref_data + "\n", fontsize=title_size, fontweight="bold", loc="left"
            )
        ax.set_title(
            "vs. " + test_datas[c],
            fontsize=title_size,
            fontweight="bold",
            loc="right",
            pad=3.0,
        )

        # Colorbars.
        if c == 0:
            c_ax = fig.add_subplot(gs_1[r])
            cb = fig.colorbar(im, cax=c_ax)
            cb.set_label(label="Abs(corr.)", fontsize=title_size, fontweight="bold")
            cb.ax.tick_params(labelsize=label_size)

# Optional saving.
if savefigs == True:
    fig.savefig(
        "/home/sotpapad/Codes/cho_2017/comps_cors{}{}.pdf".format(
            fooof_save_str, threshold_str
        ),
        dpi=dpi,
        facecolor="w",
        edgecolor="w",
    )
elif savefigs == True:
    plt.show()
