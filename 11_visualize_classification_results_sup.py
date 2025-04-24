import numpy as np
import matplotlib.pyplot as plt

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

savefigs = True

plot_format = "pdf"  # "pdf", "png"


# ----- #
# Dataset selection.
data = "cho2017"  # "cho2017"

# Mode.
mode = "cluster"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

datasets = []
datasets_std = []
subjects = []
titles_str = []

if data == "cho2017":
    variables_path = "{}cho_2017/variables.json".format(basepath)
    title_str = "Cho 2017"

titles_str.append(title_str)


# ----- #
# Loading of dataset-specific variables.
experimental_vars = load_exp_variables(json_filename=variables_path)

savepath = experimental_vars["dataset_path"]

subs = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
if data == "cho2017":
    # Some subjects are not included in the dataset.
    subs = np.delete(np.array(subs), [31, 45, 48]).tolist()


# ----- #
# Loading of decoding results.
results = {
    "bursts_pca": np.load(
        savepath
        + "mean_{}_stratified_{}_bursts_pca{}{}.npy".format(
            metric, band, fooof_save_str, hspace_str
        )
    ),
    "burst_rate": np.load(
        savepath + "mean_{}_{}_rate_simple{}.npy".format(metric, band, fooof_save_str)
    ),
    "burst_features": np.load(
        savepath + "mean_{}_{}{}_tf_features.npy".format(metric, band, fooof_save_str)
    ),
    "burst_amplitude": np.load(
        savepath + "mean_{}_{}{}_tf_features.npy".format(metric, band, fooof_save_str)
    ),
    "burst_volume": np.load(
        savepath + "mean_{}_{}{}_tf_volume.npy".format(metric, band, fooof_save_str)
    ),
    "power": np.load(
        savepath
        + "mean_{}_power_{}_band_simple{}.npy".format(metric, band, fooof_save_str)
    ),
}

results_std = {
    "bursts_pca": np.load(
        savepath
        + "std_{}_stratified_{}_bursts_pca{}{}.npy".format(
            metric, band, fooof_save_str, hspace_str
        )
    ),
    "burst_rate": np.load(
        savepath
        + "std_{}_{}_rate_simple{}.npy".format(metric, band, fooof_save_str)
    ),
    "burst_features": np.load(
        savepath + "std_{}_{}{}_tf_features.npy".format(metric, band, fooof_save_str)
    ),
    "burst_amplitude": np.load(
        savepath + "std_{}_{}{}_tf_features.npy".format(metric, band, fooof_save_str)
    ),
    "burst_volume": np.load(
        savepath + "std_{}_{}{}_tf_volume.npy".format(metric, band, fooof_save_str)
    ),
    "power": np.load(
        savepath
        + "std_{}_power_{}_band_simple{}.npy".format(metric, band, fooof_save_str)
    ),
}


# ----- #
# Figure initialization.
screen_res = [1920, 972]
dpi = 300

# Spacing of subgrids.
if data == "cho2017":
    n_rows = 5
    height = 4.5
else:
    n_rows = 11
    height = 9

n_cols = 10

if savefigs == False:
    fig = plt.figure(
        constrained_layout=False,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
    gs = fig.add_gridspec(
        nrows=1, ncols=1, left=0.02, right=0.97, top=0.98, bottom=0.00
    )
else:
    fig = plt.figure(constrained_layout=False, figsize=(7, height), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=1, ncols=1, left=0.02, right=0.97, top=0.98, bottom=0.00
    )

title_size = 5
legend_size = 7
legend_label_size = 4.5
label_size = 5
tick_size = 6
ws = 0.30

# Gridspec.
gs0 = gs[0].subgridspec(n_rows, n_cols)


# ----- #
#  Figures' hyper-parameters.

# Labels and colors.
lnc = {
    "{} burst waveforms".format(band_letter): "crimson",
    "{} burst rate".format(band_letter): "turquoise",
    "{} burst features".format(band_letter): "mediumturquoise",
    "{} burst amplitude".format(band_letter): "lightseagreen",
    "{} burst volume".format(band_letter): "teal",
    "{} band power".format(band_letter): "blueviolet",
}

labels = list(lnc.keys())
palette = list(lnc.values())

# Decoding levels.
if metric == "accuracy":
    # IF USING "ACCURACY", CHANCE LEVEL COULD BE ADDED
    radii = np.arange(40, 101, 10)
elif metric == "rocauc":
    radii = np.arange(0.4, 1.01, 0.2)

# Positioning and width of bars.
center = 2 * np.pi / len(labels)
width = center - (0.5 / len(labels))
ids = np.arange(0, len(labels), 1)
if labels[-1] != "{} band power".format(band_letter):
    ids = np.roll(ids, -1)
angles = []
for id in ids:
    angles.append((np.pi / 2) - center * id)


# ----- #
# Plots.
# 1. Per subject results.

for s, subj in enumerate(subs):
    # Occupy correct subplot.
    ax0 = fig.add_subplot(gs0[s // n_cols, s % n_cols], projection="polar")

    # Retrieve mean accuracy and std.
    this_sub = []
    this_sub_std = []

    for (key, value), (_, value_std) in zip(results.items(), results_std.items()):
        if (
            key == "bursts_pca"
            or key == "bursts_csp"
            or key == "bursts_power_mubeta_csp"
        ) and limit_hspace == False:
            this_sub.append(np.nanmax(np.nanmean(value, axis=1), axis=(1, 2))[s])
            id = np.argmax(np.nanmean(value, axis=1)[s, :, :])
            this_sub_std.append(value_std.flatten()[id])

        elif (
            key == "bursts_pca"
            or key == "bursts_csp"
            or key == "bursts_power_mubeta_csp"
        ) and limit_hspace == True:
            this_sub.append(np.nanmean(value, axis=1)[s])
            this_sub_std.append(value_std[s])

        elif key == "burst_rate" or key == "power":
            this_sub.append(np.nanmean(value, axis=(1, 2))[s])
            this_sub_std.append(value_std[s])

        else:
            this_sub.append(np.nanmean(value, axis=1)[s])
            this_sub_std.append(value_std[s])

    # Bar plot.

    # Fake bar fixes radii aliasing when the rest of the bars
    # do not have values close to 1.0
    ax0.bar(
        x=angles[0],
        height=1,
        width=width,
        bottom=0,
        color="w",
        alpha=0.0,
        linewidth=tick_size,
    )

    ax0.bar(
        x=angles,
        height=this_sub,
        yerr=this_sub_std,
        width=width,
        bottom=0,
        color=palette,
        linewidth=tick_size,
    )
    ax0.bar(
        x=0, height=radii[0], width=2 * np.pi, bottom=0, color="w", linewidth=tick_size
    )

    # Remove spines, ticks, ticklabels.
    ax0.spines[["polar"]].set_visible(False)
    ax0.set_rgrids(radii, fontweight=tick_size)
    ax0.set_xticks([])
    ax0.set_yticklabels([])

    # Accuracy texts.
    for cm, acc in enumerate(this_sub):
        acc = np.around(acc, 2)
        ax0.text(
            x=angles[cm], y=radii[-1], s="{0:.2f}".format(acc), fontsize=label_size
        )

    # Subject text.
    ax0.text(
        x=angles[0] - 1,
        y=0,
        s="S{}".format(subj),
        fontweight="bold",
        fontsize=title_size,
    )

# Optional saving.
if savefigs == True:
    fig_name = savepath + "decoding_scores_sup_{}{}.{}".format(
        data, fooof_save_str, plot_format
    )
    fig.savefig(fig_name, dpi=dpi, facecolor="w", edgecolor="w")
else:
    plt.show()
