"""
Visualization of components and groups hyper-parameter space.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.stats import binom

from help_funcs import load_exp_variables


# ----- #
# Data loading.
data = "zhou2016"  # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014"

# Mode.
mode = "cluster"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

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

basic_vars = load_exp_variables(json_filename=variables_path)
subjects = np.arange(1, basic_vars["n_subjects"] + 1, 1).tolist()
savepath = basic_vars["dataset_path"]

stratification = True

solver = "pca"  # "pca", "csp"
metric = "rocauc"  # "rocauc", "accuracy"

if stratification == False:
    mean_accs = np.load(savepath + "mean_{}_bursts_{}.npy".format(metric, solver))
    std_accs = np.load(savepath + "std_{}_bursts_{}.npy".format(metric, solver))
else:
    mean_accs = np.load(
        savepath + "mean_{}_stratified_bursts_{}.npy".format(metric, solver)
    )
    std_accs = np.load(
        savepath + "std_{}_stratified_bursts_{}.npy".format(metric, solver)
    )


# ------ #
# Figures initialization.
screen_res = [1920, 972]
dpi = 96

fig1 = plt.figure(
    constrained_layout=False,
    figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
    dpi=dpi,
)
gs1 = fig1.add_gridspec(
    nrows=2, ncols=2, hspace=0.10, left=0.05, right=0.90, top=0.95, bottom=0.05
)

fig2 = plt.figure(
    constrained_layout=False,
    figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
    dpi=dpi,
)
gs2 = fig2.add_gridspec(
    nrows=1,
    ncols=2,
    wspace=0.20,
    left=0.05,
    right=0.95,
    top=0.95,
    bottom=0.10,
    width_ratios=[0.98, 0.02],
)
gs20 = gs2[0].subgridspec(2, 2)
gs21 = gs2[1].subgridspec(1, 1)

fig3 = plt.figure(
    constrained_layout=False,
    figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
    dpi=dpi,
)
gs3 = fig3.add_gridspec(
    nrows=1,
    ncols=2,
    wspace=0.20,
    left=0.05,
    right=0.95,
    top=0.95,
    bottom=0.10,
    width_ratios=[0.97, 0.03],
)
gs30 = gs3[0].subgridspec(1, 1)
gs31 = gs3[1].subgridspec(1, 1)

xpos = np.arange(1, mean_accs.shape[-1] + 1, 1)
ypos = np.arange(2, mean_accs.shape[-2] + 2, 1)
xpos_3d, ypos_3d = np.meshgrid(xpos, ypos)

ones = np.ones((mean_accs.shape[-2], mean_accs.shape[-1]))
alpha = 0.05


# Plot.
for i in range(len(subjects)):
    if i == 0:
        ax1 = fig1.add_subplot(gs1[0, 0], projection="3d")
        ax2 = fig2.add_subplot(gs20[0, 0])
        ax21 = fig2.add_subplot(gs21[0])
        ax3 = fig3.add_subplot(gs30[0])
        ax31 = fig3.add_subplot(gs31[0])
    elif i == 1:
        ax1 = fig1.add_subplot(gs1[0, 1], projection="3d")
        ax2 = fig2.add_subplot(gs20[0, 1])
    elif i == 2:
        ax1 = fig1.add_subplot(gs1[1, 0], projection="3d")
        ax2 = fig2.add_subplot(gs20[1, 0])
    elif i == 3:
        ax1 = fig1.add_subplot(gs1[1, 1], projection="3d")
        ax2 = fig2.add_subplot(gs20[1, 1])

    # Data.
    if stratification == False:
        data = mean_accs[i, :, :]
    else:
        data = np.mean(mean_accs[i, :, :, :], axis=0)

    # Chance levels.
    if metric == "accuracy":
        ntrials = np.load(savepath + "sub_{}/ntrials.npy".format(i + 1))[0]
        onot = ntrials[0] + ntrials[1]
        rnot = ntrials[2] + ntrials[3]
        dnot = onot - rnot
        chance = binom.ppf(1 - alpha, rnot, 1 / 2) * 100 / (rnot)
        cb_values = [40, 50, np.around(np.max(data)) + 3]
        blabel = "Classification accuracy (%)"
    elif metric == "rocauc":
        chance = 0.5
        cb_values = [0.4, 0.5, np.around(np.max(data), decimals=2) + 0.03]
        blabel = "Score"

    # Surface plots.
    colornorm = colors.TwoSlopeNorm(
        vmin=cb_values[0], vcenter=cb_values[1], vmax=cb_values[2]
    )
    ax1.plot_surface(xpos_3d, ypos_3d, data, norm=colornorm, cmap="cool")
    ax1.plot_surface(xpos_3d, ypos_3d, ones * chance, color="b", alpha=0.2)

    if i == 0:
        im = ax2.imshow(
            data, aspect="auto", origin="lower", norm=colornorm, cmap="cool"
        )
        cb = fig2.colorbar(im, cax=ax21)
        cb.set_label(label=blabel, size=12)

        im1 = ax3.imshow(
            data, aspect="auto", origin="lower", norm=colornorm, cmap="cool"
        )
        cb1 = fig3.colorbar(im1, cax=ax31, extend="both")
        cb1.set_label(label=blabel, size=14)
        cb1.ax.tick_params(length=10, width=2, labelsize=12)

    else:
        ax2.imshow(data, aspect="auto", origin="lower", norm=colornorm, cmap="cool")

    # Heatmap ticks and annotations.
    ax2.set_xticks(np.arange(len(xpos)), labels=xpos)
    ax2.set_yticks(np.arange(len(ypos)), labels=ypos)
    ax2.set_xticks(np.arange(len(xpos) + 1) - 0.5, minor=True)
    ax2.set_yticks(np.arange(len(ypos) + 1) - 0.5, minor=True)
    ax2.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax2.spines[:].set_visible(False)
    ax2.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
    for k in range(len(xpos)):
        for l in range(len(ypos)):
            if np.around(data[l, k], decimals=2) >= chance:
                text = ax2.text(
                    k,
                    l,
                    str(np.around(data[l, k], decimals=2)),
                    ha="center",
                    va="center",
                    color="w",
                )
            else:
                text = ax2.text(
                    k,
                    l,
                    str(np.around(data[l, k], decimals=2)),
                    ha="center",
                    va="center",
                    color="k",
                )

    if i == 0:
        ax3.set_xticks(np.arange(len(xpos)), labels=xpos, fontsize=14)
        ax3.set_yticks(np.arange(len(ypos)), labels=ypos, fontsize=14)
        ax3.set_xticks(np.arange(len(xpos) + 1) - 0.5, minor=True)
        ax3.set_yticks(np.arange(len(ypos) + 1) - 0.5, minor=True)
        ax3.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax3.spines[:].set_visible(False)
        ax3.tick_params(which="minor", bottom=False, top=False, left=False, right=False)
        ax3.set_xticklabels(np.arange(1, len(xpos) + 1), fontsize=14)
        ax3.set_yticklabels(np.arange(2, len(ypos) + 2), fontsize=14)
        for k in range(len(xpos)):
            for l in range(len(ypos)):
                if np.around(data[l, k], decimals=2) >= chance:
                    text1 = ax3.text(
                        k,
                        l,
                        str(np.around(data[l, k], decimals=2)),
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=14,
                    )
                else:
                    text1 = ax3.text(
                        k,
                        l,
                        str(np.around(data[l, k], decimals=2)),
                        ha="center",
                        va="center",
                        color="k",
                        fontsize=14,
                    )

    # z axis normalization.
    if metric == "accuracy":
        ax1.set_zlim([40, np.around(np.max(data) + 3)])
    elif metric == "rocauc":
        ax1.set_zlim([0.4, np.around(np.max(data) + 0.03)])

    # Camera.
    ax1.azim = -45
    ax1.elev = 15

    # Legends and titles.
    ax1.set_xlabel("# components")
    ax1.set_ylabel("# classification features")
    ax1.set_zlabel(blabel)
    ax1.set_title("Subject {}".format(i + 1))

    if i == 0 or i == 2:
        ax2.set_ylabel("# classification features", fontsize=12)
    if i >= 2:
        ax2.set_xlabel("# components", fontsize=12)
    ax2.set_title("Subject {}".format(i + 1))

    ax3.set_xlabel("# components", fontsize=14)
    ax3.set_ylabel("# classification features", fontsize=14)
    if i == 0:
        ax3.set_title("Subject {}".format(i + 1), fontsize=14)

    # Print info.
    print("Subject {}".format(i + 1))
    print(
        "Maximum classification accuracy: {}%".format(
            np.around(np.max(data), decimals=2)
        )
    )
    print("Chance level: {}%".format(np.around(chance, decimals=2)))
    print(
        "Number of components: {}".format(
            np.unravel_index(np.argmax(data), data.shape)[1] + 1
        )
    )
    print(
        "Number of features: {}".format(
            np.unravel_index(np.argmax(data), data.shape)[0] + 2
        )
    )
    print("\n")

plt.show()
