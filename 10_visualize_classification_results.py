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
datas = ["zhou2016", "2014004", "2014001",] # ["munichmi", "weibo2014", "cho2017"] #

# Mode.
mode = "local"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

datasets = []
datasets_std = []
subjects = []
titles_str = []

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
        title_str = "Munich MI (Grosse-Wentrup)"
    elif data == "cho2017":
        variables_path = "{}cho_2017/variables.json".format(basepath)
        title_str = "Cho 2017"
    elif data == "weibo2014":
        variables_path = "{}weibo_2014/variables.json".format(basepath)
        title_str = "Weibo 2014"

    titles_str.append(title_str)

    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    savepath = experimental_vars["dataset_path"]

    subs = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
    if data == "cho2017":
        # Some subjects are not included in the dataset.
        subs = np.delete(np.array(subs), [31, 45, 48]).tolist()
    subjects.append(subs)

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
            savepath
            + "mean_{}_{}_rate_simple{}.npy".format(metric, band, fooof_save_str)
        ),
        "burst_features": np.load(
            savepath
            + "mean_{}_{}{}_tf_features.npy".format(metric, band, fooof_save_str)
        ),
        "burst_amplitude": np.load(
            savepath
            + "mean_{}_{}{}_tf_amplitude.npy".format(metric, band, fooof_save_str)
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
            savepath
            + "std_{}_{}{}_tf_features.npy".format(metric, band, fooof_save_str)
        ),
        "burst_amplitude": np.load(
            savepath
            + "std_{}_{}{}_tf_amplitude.npy".format(metric, band, fooof_save_str)
        ),
        "burst_volume": np.load(
            savepath + "std_{}_{}{}_tf_volume.npy".format(metric, band, fooof_save_str)
        ),
        "power": np.load(
            savepath
            + "std_{}_power_{}_band_simple{}.npy".format(metric, band, fooof_save_str)
        ),
    }

    datasets.append(results)
    datasets_std.append(results_std)


# ----- #
# Figure initialization.
screen_res = [1920, 972]
dpi = 300

# Spacing of subgrids.
if len(datasets) == 1:
    lm = 0.05
    hs = -0.5
elif len(datasets) == 2:
    lm = 0.02
    hs = 0.0
else:
    lm = 0.02
    hs = 0.20

if savefigs == False:
    fig = plt.figure(
        constrained_layout=False,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
    gs = fig.add_gridspec(
        nrows=len(datasets),
        ncols=2,
        wspace=0.15,
        left=lm,
        right=0.95,
        top=0.95,
        bottom=0.05,
        width_ratios=[0.35, 0.65],
    )
    title_size = 5
    legend_size = 5
    legend_label_size = 4.5
    label_size = 4
    tick_size = 4
    ws = 0.30

else:
    fig = plt.figure(constrained_layout=False, figsize=(7, 2.25 * len(datas)), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=len(datasets),
        ncols=2,
        wspace=0.05,
        left=lm,
        right=0.95,
        top=0.95,
        bottom=0.05,
        width_ratios=[0.35, 0.65],
    )
    title_size = 6
    legend_size = 7
    legend_label_size = 5.5
    label_size = 5
    tick_size = 6
    ws = 0.30

# Gridspec.
gs00 = gs[0].subgridspec(1, 1)

if len(datasets) == 1:
    gs1 = [gs[1].subgridspec(2, 5, wspace=ws, hspace=hs)]
    ax00 = fig.add_subplot(gs00[0], projection="polar")
    axes0 = [ax00]

elif len(datasets) >= 2:
    gs01 = gs[0, 1].subgridspec(2, 5, wspace=ws, hspace=hs)
    gs10 = gs[1, 0].subgridspec(1, 1)
    gs11 = gs[1, 1].subgridspec(2, 5, wspace=ws, hspace=hs)
    ax00 = fig.add_subplot(gs00[0], projection="polar")
    ax10 = fig.add_subplot(gs10[0], projection="polar")
    axes0 = [ax00, ax10]
    gs1 = [gs01, gs11]

    if len(datasets) >= 3:
        gs20 = gs[2, 0].subgridspec(1, 1)
        gs21 = gs[2, 1].subgridspec(2, 5, wspace=ws, hspace=hs)
        ax20 = fig.add_subplot(gs20[0], projection="polar")
        axes0.append(ax20)
        gs1.append(gs21)

        if len(datasets) == 4:
            gs30 = gs[3, 0].subgridspec(1, 1)
            gs31 = gs[3, 1].subgridspec(2, 5, wspace=ws, hspace=hs)
            ax30 = fig.add_subplot(gs30[0], projection="polar")
            axes0.append(ax30)
            gs1.append(gs31)


# ----- #
# Figures' hyper-parameters.

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
# 1. Average across subjects.

# Dataset-average results.
asm = []
ass = []

for results, results_std in zip(datasets, datasets_std):
    # Retrieve mean accuracy and std.
    across_subjects = []
    across_subjects_std = []

    for (key, value), (_, value_std) in zip(results.items(), results_std.items()):
        if (
            key == "bursts_pca"
            or key == "bursts_csp"
            or key == "bursts_power_mubeta_csp"
        ) and limit_hspace == False:
            across_subjects.append(
                np.nanmean(np.nanmax(np.nanmean(value, axis=1), axis=(1, 2)))
            )
            stds = []
            for v, vs in zip(value, value_std):
                id = np.argmax(v)
                stds.append(vs.flatten()[id])
            across_subjects_std.append(np.nanmean(stds))

        elif (
            key == "bursts_pca"
            or key == "bursts_csp"
            or key == "bursts_power_mubeta_csp"
        ) and limit_hspace == True:
            across_subjects.append(np.nanmean(value))
            across_subjects_std.append(np.nanmean(value_std))

        else:
            across_subjects.append(np.nanmean(value))
            across_subjects_std.append(np.nanmean(value_std))

    asm.append(across_subjects)
    ass.append(across_subjects_std)

for d, (across_subjects, ax) in enumerate(zip(asm, axes0)):
    # All results per dataset.
    accs = []
    stds = []

    for cm, acc in enumerate(across_subjects):
        accs.append(np.around(acc, 2))
        stds.append(np.around(across_subjects_std[cm], 2))

        # Bar plot.

        # Fake bar fixes radii aliasing when the rest of the bars
        # do not have values close to 1.0
        ax.bar(
            x=angles[0],
            height=1,
            width=width,
            bottom=0,
            color="w",
            alpha=0.0,
            linewidth=tick_size,
        )

        if d == 0:
            # Labels only for the first dataset.
            ax.bar(
                x=angles[cm],
                height=across_subjects[cm],
                yerr=stds[cm],
                width=width,
                bottom=0,
                color=palette[cm],
                label=labels[cm],
                linewidth=tick_size,
            )
        else:
            ax.bar(
                x=angles[cm],
                height=across_subjects[cm],
                yerr=stds[cm],
                width=width,
                bottom=0,
                color=palette[cm],
                linewidth=tick_size,
            )

        # Set minimum visible decoding level.
        ax.bar(x=0, height=radii[0], width=2 * np.pi, bottom=0, color="w")

        # Accuracy text.
        ax.text(
            x=angles[cm], y=radii[-1], s="{0:.2f}".format(accs[cm]), fontsize=label_size
        )

    # Remove spines, ticks, ticklabels.
    ax.spines[["polar"]].set_visible(False)
    ax.set_rgrids(radii, fontweight=tick_size)
    ax.yaxis.grid(linewidth=1.0)
    ax.set_xticks([])
    ax.set_yticklabels([])

    # Titles.
    ax.set_title(titles_str[d], fontsize=label_size, loc="left", fontweight="bold")


# 2. Per subject results.
for r, (results, results_std, gs) in enumerate(zip(datasets, datasets_std, gs1)):
    # Select some subjects from "Cho 2017" and "PhysionetMI" datasets.
    if datas[r] == "cho2017" or datas[r] == "physionet":
        # Choose best subjects with respect to burst waveforms
        # classification scores.
        if limit_hspace == False:
            order = np.argsort(
                np.nanmax(np.nanmean(results["bursts_pca"], axis=1), axis=(1, 2))
            )
        elif limit_hspace == True:
            order = np.argsort(np.nanmean(results["bursts_pca"], axis=1))
        best_subjects = np.array(subjects[r])[order[-10:]]

        for s, subj in enumerate(best_subjects):
            s_orig = int(np.where(subjects[r] == subj)[0])

            # Occupy correct subplot.
            half = int(np.ceil(len(best_subjects) / 2))
            if s < half:
                ax1 = fig.add_subplot(gs[0, s], projection="polar")
            else:
                ax1 = fig.add_subplot(gs[1, s - half], projection="polar")

            # Retrieve mean accuracy and std.
            this_sub = []
            this_sub_std = []

            for (key, value), (_, value_std) in zip(
                results.items(), results_std.items()
            ):
                if (
                    key == "bursts_pca"
                    or key == "bursts_csp"
                    or key == "bursts_power_mubeta_csp"
                ) and limit_hspace == False:
                    this_sub.append(
                        np.nanmax(np.nanmean(value, axis=1), axis=(1, 2))[s_orig]
                    )
                    id = np.argmax(np.nanmean(value, axis=1)[s_orig, :, :])
                    this_sub_std.append(value_std.flatten()[id])

                elif (
                    key == "bursts_pca"
                    or key == "bursts_csp"
                    or key == "bursts_power_mubeta_csp"
                ) and limit_hspace == True:
                    this_sub.append(np.nanmean(value, axis=1)[s_orig])
                    this_sub_std.append(value_std[s_orig])

                elif key == "burst_rate" or key == "power":
                    this_sub.append(np.nanmean(value, axis=(1, 2))[s_orig])
                    this_sub_std.append(value_std[s_orig])

                else:
                    this_sub.append(np.nanmean(value, axis=1)[s_orig])
                    this_sub_std.append(value_std[s_orig])

            # Bar plot.

            # Fake bar fixes radii aliasing when the rest of the bars
            # do not have values close to 1.0.
            ax1.bar(
                x=angles[0],
                height=1,
                width=width,
                bottom=0,
                color="w",
                alpha=0.0,
                linewidth=tick_size,
            )

            ax1.bar(
                x=angles,
                height=this_sub,
                yerr=this_sub_std,
                width=width,
                bottom=0,
                color=palette,
                linewidth=tick_size,
            )
            ax1.bar(
                x=0,
                height=radii[0],
                width=2 * np.pi,
                bottom=0,
                color="w",
                linewidth=tick_size,
            )

            # Remove spines, ticks, ticklabels.
            ax1.spines[["polar"]].set_visible(False)
            ax1.set_rgrids(radii, fontweight=tick_size)
            ax1.yaxis.grid(linewidth=1.0)
            ax1.set_xticks([])
            ax1.set_yticklabels([])

            # Accuracy texts.
            for cm, acc in enumerate(this_sub):
                acc = np.around(acc, 2)
                ax1.text(
                    x=angles[cm],
                    y=radii[-1],
                    s="{0:.2f}".format(acc),
                    fontsize=label_size,
                )

            # Subject text.
            ax1.text(
                x=angles[0] - 1,
                y=0,
                s="S{}".format(subj),
                fontweight="bold",
                fontsize=title_size,
            )

    # Rest of datasets.
    else:
        for s, subj in enumerate(subjects[r]):
            # Occupy correct subplot.
            half = int(np.ceil(len(subjects[r]) / 2))
            if s < half:
                ax1 = fig.add_subplot(gs[0, s], projection="polar")
            else:
                ax1 = fig.add_subplot(gs[1, s - half], projection="polar")

            # Retrieve mean accuracy and std.
            this_sub = []
            this_sub_std = []

            for (key, value), (_, value_std) in zip(
                results.items(), results_std.items()
            ):
                if (
                    key == "bursts_pca"
                    or key == "bursts_csp"
                    or key == "bursts_power_mubeta_csp"
                ) and limit_hspace == False:
                    this_sub.append(
                        np.nanmax(np.nanmean(value, axis=1), axis=(1, 2))[s]
                    )
                    arg = np.argmax(np.nanmean(value, axis=1)[s, :, :])
                    this_sub_std.append(value_std.flatten()[arg])

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
            ax1.bar(
                x=angles[0],
                height=1,
                width=width,
                bottom=0,
                color="w",
                alpha=0.0,
                linewidth=tick_size,
            )

            ax1.bar(
                x=angles,
                height=this_sub,
                yerr=this_sub_std,
                width=width,
                bottom=0,
                color=palette,
                linewidth=tick_size,
            )
            ax1.bar(
                x=0,
                height=radii[0],
                width=2 * np.pi,
                bottom=0,
                color="w",
                linewidth=tick_size,
            )

            # Remove spines, ticks, ticklabels.
            ax1.spines[["polar"]].set_visible(False)
            ax1.set_rgrids(radii, fontweight=tick_size)
            ax1.yaxis.grid(linewidth=1.0)
            ax1.set_xticks([])
            ax1.set_yticklabels([])

            # Accuracy texts.
            for cm, acc in enumerate(this_sub):
                acc = np.around(acc, 2)
                ax1.text(
                    x=angles[cm],
                    y=radii[-1],
                    s="{0:.2f}".format(acc),
                    fontsize=label_size,
                )

            # Subject text.
            ax1.text(
                x=angles[0] - 1,
                y=0,
                s="S{}".format(subj),
                fontweight="bold",
                fontsize=title_size,
            )

# Legend.
if datas[0] == "zhou2016":
    leg_cols = 2
    fig.legend(
        frameon=False,
        title="Feature extraction pipeline",
        title_fontsize=legend_size,
        fontsize=legend_label_size,
        ncol=leg_cols,
        alignment="left",
    )

# Optional saving.
if savefigs == True:
    fig_name = savepath + "decoding_scores{}{}.{}".format(
        hspace_str, fooof_save_str, plot_format
    )
    fig.savefig(fig_name, dpi=dpi, facecolor="w", edgecolor="w")
else:
    plt.show()
