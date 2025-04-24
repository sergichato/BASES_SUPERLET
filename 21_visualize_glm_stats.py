import numpy as np
import matplotlib.pyplot as plt


# ----- #
# Variables.
savepath = "/home/sotpapad/Codes/dreyer_2023/"

savefigs = True  # True, False

classification_mode = "incremental"   # "incremental", "sliding"
if classification_mode == "incremental":
    clm_str = "_tr"
elif classification_mode == "sliding":
    clm_str = "_sl"

plot_format = "pdf"  # "pdf", "png"


# ----- #
# Data.
if classification_mode == "incremental":
    model_acc = {
        "emmean": np.array([0.832, 0.755, 0.827, 0.734, 0.819]),
        "lower_cl": np.array([0.766, 0.690, 0.761, 0.668, 0.753]),
        "upper_cl": np.array([0.898, 0.821, 0.893, 0.800, 0.885]),
        "pval": np.array([0.0001, 0.9691, 0.0001, 0.3873]),
        "y_label": "max (score)",
    }

    model_time = {
        "emmean": np.array([2.26, 3.04, 2.96, 2.50, 2.44]),
        "lower_cl": np.array([1.36, 2.13, 2.06, 1.59, 1.54]),
        "upper_cl": np.array([3.17, 3.94, 3.87, 3.40, 3.35]),
        "pval": np.array([0.0001, 0.0001, 0.3034, 0.5703]),
        "y_label": "$Time_{max (score)} (s)$",
    }

    model_itr = {
        #"emmean": np.array([-1.19, -2.43, -1.64, -2.32, -1.45]),
        "emmean": np.array([0.6443, 0.0554, 0.3531, 0.0667, 0.5241]),
        #"lower_cl": np.array([-1.79, -3.03, -2.24, -2.92, -2.05]),
        "lower_cl": np.array([0.437, -0.152, 0.146, 0.140, 0.317]),
        #"upper_cl": np.array([-0.587, -1.827, -1.037, -1.722, -0.844]),
        "upper_cl": np.array([0.852, 0.262, 0.560, 0.274, 0.732]),
        "pval": np.array([0.0001, 0.0001, 0.0001, 0.0298]),
        "y_label": "log (max (ITR)) (a.u.)",
    }

    model_itr_time = {
        #"emmean": np.array([-0.677, -0.168, -0.341, -0.497, -0.542]),
        "emmean": np.array([0.827, 1.316, 1.182, 1.064, 0.945]),
        #"lower_cl": np.array([-0.894, -0.384, -0.556, -0.713, -0.760]),
        "lower_cl": np.array([0.624, 1.114, 0.981, 0.863, 0.742]),
        #"upper_cl": np.array([-0.4602, 0.0476, -0.1262, -0.2812, -0.3248]),
        "upper_cl": np.array([1.03, 1.52, 1.38, 1.27, 1.15]),
        "pval": np.array([0.0001, 0.0010, 0.2329, 0.5329]), # < =
        "y_label": "$log (time_{max (ITR)}) (a.u.$)",
    }

elif classification_mode == "sliding":
    model_acc = {
        "emmean": np.array([0.816, 0.753, 0.810, 0.753, 0.818]),
        "lower_cl": np.array([0.766, 0.703, 0.760, 0.703, 0.768]),
        "upper_cl": np.array([0.866, 0.803, 0.860, 0.803, 0.868]),
        "pval": np.array([0.0001, 0.8748, 0.0001, 0.9936]),
        "y_label": "max (score)",
    }

    model_time = {
        "emmean": np.array([2.03, 2.45, 2.19, 2.49, 2.22]),
        "lower_cl": np.array([1.79, 2.22, 1.95, 2.26, 1.99]),
        "upper_cl": np.array([2.26, 2.68, 2.42, 2.73, 2.46]),
        "pval": np.array([0.0009, 0.5840, 0.0002, 0.3961]), # =
        "y_label": "$Time_{max (score)} (s)$",
    }

    model_itr = {
        #"emmean": np.array([-0.977, -1.808, -1.341, -1.722, -1.113]),
        "emmean": np.array([0.697, 0.170, 0.413, 0.206, 0.533]),
        #"lower_cl": np.array([-1.34, -2.17, -1.70, -2.08, -1.47]),
        "lower_cl": np.array([0.51841, -0.00791, 0.26565, 0.02767, 0.35408]),
        #"upper_cl": np.array([-0.619, -1.451, -0.984, -1.365, -0.755]),
        "upper_cl": np.array([0.876, 0.348, 0.590, 0.384, 0.712]),
        "pval": np.array([0.0001, 0.0001, 0.0001, 0.4559]), # < = <
        "y_label": "log (max (ITR)) (a.u.)",
    }

    model_itr_time = {
        #"emmean": np.array([-1.275, -0.906, -0.912, -1.059, -1.118]),
        "emmean": np.array([0.575, 0.840, 0.796, 0.743, 0.701]),
        #"lower_cl": np.array([-1.61, -1.24, -1.24, -1.39, -1.45]),
        "lower_cl": np.array([0.413, 0.679, 0.636, 0.582, 0.539]),
        #"upper_cl": np.array([-0.942, -0.573, -0.581, -0.727, -0.784]),
        "upper_cl": np.array([0.736, 1.001, 0.957, 0.904, 0.863]),
        "pval": np.array([0.0183, 0.0207, 0.3756, 0.6937]),
        "y_label": "$log (time_{max (ITR)}) (a.u.$)",
    }

models_sl = [model_acc, model_time, model_itr, model_itr_time]


# ----- #
# Figure initialization.
features = [
    "Convolution & CSP",
    "Beta band filter\n& CSP (15-30 Hz)",
    "Mu-beta band filter\n& CSP (6-30 Hz)",
    "Beta band filter bank\n& CSP (15-30 Hz)",
    "Mu-beta band filter\nbank & CSP (6-30 Hz)",
]

colors = [
    "orangered",
    "goldenrod",
    "mediumorchid",
    "goldenrod",
    "mediumorchid",
]

hatches = [
    "",
    "",
    "",
    "///",
    "///"
]

fig = plt.figure(constrained_layout=False, figsize=(7, 3), dpi=300)
gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        wspace=0.30,
        hspace=0.50,
        left=0.15,
        right=0.98,
        top=0.90,
        bottom=0.15,
    )

gs00 = gs[0,0].subgridspec(1, 1)
gs01 = gs[0,1].subgridspec(1, 1)
gs02 = gs[1,0].subgridspec(1, 1)
gs03 = gs[1,1].subgridspec(1, 1)

ax00 = fig.add_subplot(gs00[0])
ax01 = fig.add_subplot(gs01[0])
ax10 = fig.add_subplot(gs02[0])
ax11 = fig.add_subplot(gs03[0])

all_axes = [ax00, ax10, ax01, ax11]

width = 0.5


# ----- #
# Subplots.
for i, (model, ax) in enumerate(zip(models_sl, all_axes)):

    # Std.
    #if i <= 1:
    err = [
        np.abs(model["emmean"] - model["lower_cl"]),
        np.abs(model["upper_cl"] - model["emmean"]),
    ]
    #else:
    #    err = [
    #        np.exp(model["emmean"]) - np.exp(model["lower_cl"]),
    #        np.exp(model["upper_cl"]) - np.exp(model["emmean"]),
    #    ]

    # Barplot.
    ax.barh(
        features,
        model["emmean"],# if i <= 1 else np.exp(model["emmean"]),
        height=width,
        xerr=err,
        align="center",
        color=colors,
        hatch=hatches,
        zorder=10,
    )

    # Ticks, labels.
    ax.set_yticklabels(features, fontsize=4)
    if i <= 1:
        ax.set_ylabel("Feture extraction pipeline", fontsize=6)
    if i >= 2:
        ax.set_yticklabels([""])
        #ax.yaxis.tick_right()
    
    if i == 0:
        ax.set_xlim([0.4, 1.01])
    
    ax.invert_yaxis()
    
    ax.set_xlabel(model["y_label"], fontsize=6)
    ax.xaxis.set_tick_params(labelsize=4)

    if i <= 1:
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.spines[["top", "right"]].set_visible(False)
    
    ax.grid(True, axis="x", zorder=0)

# Optional saving.
if savefigs == True:
    plt.savefig(savepath + "stats_{}.{}".format(classification_mode, plot_format))
else:
    plt.show()