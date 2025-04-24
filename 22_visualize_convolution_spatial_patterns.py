import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from os.path import join, dirname

from moabb.datasets import (
    Zhou2016,
    BNCI2014004,
    BNCI2014001,
    MunichMI,
    Weibo2014,
    Cho2017,
)

from mne.decoding import CSP
from mne.viz import plot_topomap

from preprocess import load_sub, apply_preprocessing
from help_funcs import load_exp_variables
from burst_space import BurstSpace


# ----- #
# Selection of classification hyperparameters.
data = "zhou2016"       # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014", "dreyer2023"
kernel = "waveforms"    # "drm_components", "waveforms"
band = "beta"           # "beta", "mu"
remove_fooof = False    # True, False

trials_fraction = 1.0

filter_bank = [
    [15, 30],
    [6, 15],
    [6, 30],
]

csp_comp_num = 2
csp_order = "alternate"  # "mutual_info", "alternate"
csp_order_txt = "mi" if csp_order == "mutual_info" else "al"
csp = CSP(
    n_components=csp_comp_num,
    reg=None,
    log=True,
    transform_into="average_power",
    component_order=csp_order,
)

# Recordings scaling.
scalings = dict(eeg=1e6, grad=1e13, mag=1e15)

# Components to use.
cta = [2, 3, 4, 5, 6, 7, 8]

# Number of groups per component.
n_groups = 7

# Type of waveforms to be retained for the analysis.
output_waveforms = "extrema"  # "all", "mid_extrema", "extrema"
n_comps = 3

# Selection of amplitude and CSP averaging using all previous
# time points, or short time window (+/- 100ms).
power_patterns = "averaged"  # "averaged", "windowed"

# Mode.
mode = "local"  # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/"

# Figure hyperpearms.
savefigs = True
plot_format = "pdf"  # "pdf", "png"

left_hand_img = mpimg.imread(join(dirname(__file__), "./left_hand.png"))
right_hand_img = mpimg.imread(join(dirname(__file__), "./right_hand.png"))


# ----- #
# Dataset selection.
if data == "zhou2016":
    dataset = Zhou2016()
    dataset_name = "Zhou 2016"
    variables_path = "{}zhou_2016/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
elif data == "2014004":
    dataset = BNCI2014004()
    dataset_name = "BNCI 2014-004"
    variables_path = "{}2014_004/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
elif data == "2014001":
    dataset = BNCI2014001()
    dataset_name = "BNCI 2014-001"
    variables_path = "{}2014_001/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
elif data == "munichmi":
    dataset = MunichMI()
    dataset_name = "MunichMI (Grosse-Wentrup 2009)"
    variables_path = "{}munichmi/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = 24.8
    noise_wins = [1, 0.5]
elif data == "cho2017":
    dataset = Cho2017()
    dataset_name = "Cho 2017"
    variables_path = "{}cho_2017/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = 60.0
    noise_wins = [10, 5]
elif data == "weibo2014":
    dataset = Weibo2014()
    dataset_name = "Weibo 2014"
    variables_path = "{}weibo_2014/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
elif data == "dreyer2023":
    dataset = Dreyer2023(basepath=basepath + "dreyer_2023/")
    dataset_name = "Dreyer 2023"
    variables_path = "{}dreyer_2023/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None


# ----- #
# Loading of dataset-specific variables.
experimental_vars = load_exp_variables(json_filename=variables_path)

savepath = experimental_vars["dataset_path"]

subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
if data == "cho2017":
    # Some subjects are not included in the dataset.
    subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

channels = None
channel_ids = experimental_vars["channel_ids"]

# Time period of task.
tmin = experimental_vars["tmin"]
tmax = experimental_vars["tmax"]
exp_time_periods = experimental_vars["exp_time_periods"]
sfreq = experimental_vars["sfreq"]

exp_time = np.linspace(tmin, tmax, int((np.abs(tmax - tmin)) * sfreq) + 1)
exp_time = np.around(exp_time, decimals=3)

base_start = -0.5
base_id = int(np.where(exp_time == base_start)[0])
time_points = [0.0, 0.5, 1.0, 2.0, exp_time_periods[2]]
time_ids = [int(np.where(exp_time == tp)[0]) for tp in time_points]

# Time windows.
window_length = experimental_vars["bin_dt"]
samples_step = np.around(exp_time[1] - exp_time[0], decimals=4)
window_samples = int(window_length * 5 / samples_step)


# ----- #
# Figures.
for s, subject in enumerate(subjects[:1]):
    print("Estimating spatial distribution of power for subject {}...".format(subject))

    # ----- #
    # Figure initialization.
    dpi = 300
    screen_res = [1920, 972]

    if savefigs == False:
        fig0 = plt.figure(
            constrained_layout=False,
            figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
            dpi=dpi,
        )
        fig1 = plt.figure(
            constrained_layout=False,
            figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
            dpi=dpi,
        )
        fig2 = plt.figure(
            constrained_layout=False,
            figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
            dpi=dpi,
        )
    elif savefigs == True:
        fig0 = plt.figure(constrained_layout=False, figsize=(7, 4), dpi=dpi)
        fig1 = plt.figure(constrained_layout=False, figsize=(7, 4), dpi=dpi)
        fig2 = plt.figure(constrained_layout=False, figsize=(7, 4), dpi=dpi)

    textsizes = [6, 4, 4]

    gs0 = fig0.add_gridspec(
        nrows=7,
        ncols=7,
        wspace=0.25,
        hspace=0.40,
        left=0.02,
        right=0.93,
        top=0.90,
        bottom=0.05,
        width_ratios=[0.09, 0.18, 0.18, 0.18, 0.18, 0.18, 0.01],
        height_ratios=[0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.10],
    )

    if csp_order == "alternate":
        hr = [0.18, 0.18, 0.18, 0.18, 0.18, 0.18, 0.10]
        hr_2 = [0.18, 0.18, 0.18, 0.18, 0.18, 0.10]
    else:
        hr = [0.18, 0.18, 0.18, 0.18, 0.18, 0.18]
    gs1 = fig1.add_gridspec(
        nrows=6 if csp_order == "mutual_info" else 7,
        ncols=7,
        wspace=0.25,
        hspace=0.40,
        left=0.02,
        right=0.93,
        top=0.90,
        bottom=0.05,
        width_ratios=[0.09, 0.18, 0.18, 0.18, 0.18, 0.18, 0.01],
        height_ratios=hr,
    )
    gs2 = fig2.add_gridspec(
        nrows=6,
        ncols=7,
        wspace=0.25,
        hspace=0.40,
        left=0.02,
        right=0.93,
        top=0.90,
        bottom=0.05,
        width_ratios=[0.09, 0.18, 0.18, 0.18, 0.18, 0.18, 0.01],
        height_ratios=hr_2,
    )

    # ----- #
    # Figure data.
    # Suject-specific directory.
    sub_dir = join(savepath, "sub_{}/".format(subject))

    # Subject's raw data loading.
    print("Loading raw data...")

    if data == "physionet":
        epochs, labels, meta = load_sub(
            subject,
            dataset,
            tmin,
            tmax,
            exp_time_periods[:2],
            savepath,
            band_pass=[0, 75],
        )
    elif data == "weibo2014":
        epochs, labels, meta = load_sub(
            subject,
            dataset,
            tmin,
            tmax,
            exp_time_periods[:2],
            savepath,
            band_pass=[0, 90],
        )
    else:
        epochs, labels, meta = load_sub(
            subject, dataset, tmin, tmax, exp_time_periods[:2], savepath
        )

    # Pre-processing.
    print("Applying pre-processing...")

    # Correspondence of trials used for burst detection and
    # trials used for plotting.
    if channels == None and trials_fraction < 1.0:
        n_trials = len(labels)
        init_trials = np.arange(0, n_trials, 1)

        if data == "munichmi" or data == "cho2017":
            dummy_epochs, _, _, _, _ = apply_preprocessing(
                epochs.copy(),
                labels,
                meta,
                channels=experimental_vars["channels"],
                zapit=True,
                noise_freq=noise_freq,
                noise_wins=noise_wins,
                return_epochs=True,
            )
        else:
            dummy_epochs, _, _, _, _ = apply_preprocessing(
                epochs.copy(),
                labels,
                meta,
                channels=experimental_vars["channels"],
                zapit=zapit,
                noise_freq=noise_freq,
                noise_wins=noise_wins,
                return_epochs=True,
            )

        epochs, labels, _, info, _ = apply_preprocessing(
            epochs,
            labels,
            meta,
            channels=channels,
            zapit=zapit,
            noise_freq=noise_freq,
            noise_wins=noise_wins,
            return_epochs=True,
        )

        dummy_drop = []
        for tr, ep in enumerate(dummy_epochs.drop_log):
            if len(ep) != 0:
                dummy_drop.append(tr)
        dummy_tr_kept = np.delete(init_trials, dummy_drop)

        ep_drop = []
        for tr, ep in enumerate(epochs.drop_log):
            if len(ep) != 0:
                ep_drop.append(tr)
        tr_kept = np.delete(init_trials, ep_drop)

    else:
        epochs, labels, _, info, _ = apply_preprocessing(
            epochs,
            labels,
            meta,
            channels=channels,
            zapit=zapit,
            noise_freq=noise_freq,
            noise_wins=noise_wins,
            return_epochs=True,
        )

    # ----- #
    # Filtered data.
    all_epochs = []
    for fb in filter_bank:
        # Filtering.
        filter_band_epochs = epochs.copy().filter(fb[0], fb[1])

        # Band-passed signals.
        filter_band_epochs = filter_band_epochs.get_data()

        all_epochs.append(filter_band_epochs)

    # ----- #
    # Burst space model.
    bspace = BurstSpace(
        experimental_vars,
        subjects,
        trials_fraction=trials_fraction,
        channel_ids=channel_ids,
        remove_fooof=remove_fooof,
        band=band,
        verbose=False,
    )
    bspace.fit_transform(solver="pca", n_components=cta[-1], output="waveforms")
    drm_components, binned_waveforms, _, _ = bspace.estimate_waveforms(
        cta,
        n_groups,
        output_waveforms=output_waveforms,
        n_comps=n_comps,
    )
    if trials_fraction < 1.0:
        drm_trials = bspace.drm_trials[s]

    # Kernel selection and convolution.
    if kernel == "drm_components":
        conv_kernels = drm_components
    elif kernel == "waveforms":
        conv_kernels = []
        for comp_axis in binned_waveforms:
            for binned_waveform in comp_axis:
                conv_kernels.append(binned_waveform)

    # If needed, find correspondence between trials in burst dictionary
    # and trials actually used.
    if channels == None and trials_fraction < 1.0:
        drm_trs = np.intersect1d(tr_kept, dummy_tr_kept[drm_trials])
        temp = []
        for k in drm_trs:
            if k in tr_kept:
                temp.append(int(np.where(tr_kept == k)[0]))

        drm_trials = temp

    # Separate classes according to available labels.
    if trials_fraction < 1.0:
        # Remove trials whose data were used while creating
        # the burst space model.
        labels = np.delete(labels, drm_trials)
    lab_1 = np.where(labels == np.unique(labels)[0])[0]
    lab_2 = np.where(labels == np.unique(labels)[1])[0]

    # ----- #
    # Convolution with selected kernels.
    for ck, conv_kernel in enumerate(conv_kernels):
        if trials_fraction < 1.0:
            # Remove trials whose data were used while creating the burst space model.
            conv_kernel_data = np.delete(epochs.copy().get_data(), drm_trials, axis=0)
        else:
            conv_kernel_data = epochs.copy().get_data()

        conv_kernel_data = np.apply_along_axis(
            np.convolve, -1, conv_kernel_data, conv_kernel, mode="same"
        )

        # ----- #
        # Baseline power.
        conv_base_1 = np.mean(
            conv_kernel_data[lab_1, :, base_id : time_ids[0]] ** 2, axis=(0, 2)
        )
        conv_base_2 = np.mean(
            conv_kernel_data[lab_2, :, base_id : time_ids[0]] ** 2, axis=(0, 2)
        )

        # ----- #
        # Color normalization.
        # Power % change - Class 1.
        conv_kernel_data_1_all = np.mean(
            conv_kernel_data[lab_1, :, :] ** 2, axis=(0, 2)
        )
        conv_kernel_data_1_all = (
            (conv_kernel_data_1_all - conv_base_1) / conv_base_1 * 100
        )

        # Power % change - Class 2.
        conv_kernel_data_2_all = np.mean(
            conv_kernel_data[lab_2, :, :] ** 2, axis=(0, 2)
        )
        conv_kernel_data_2_all = (
            (conv_kernel_data_2_all - conv_base_2) / conv_base_2 * 100
        )

        vlims = [
            np.around(
                np.min([conv_kernel_data_1_all, conv_kernel_data_2_all]), decimals=0
            ),
            np.around(
                np.max([conv_kernel_data_1_all, conv_kernel_data_2_all]), decimals=0
            ),
        ]

        # Power colorbar limits.
        min_ticks = [-5, -10, -25, -50, -75, -100, -150, -200, -300, -400]
        min_tick = np.where(vlims[0] < min_ticks)[0]
        if len(min_tick) > 0:
            cb_min = min_ticks[
                min_tick[-1] + 1 if min_tick[-1] != len(min_ticks) - 1 else min_tick[-1]
            ]
        else:
            cb_min = min_ticks[0]

        max_ticks = [5, 10, 25, 50, 75, 100, 150, 200, 300, 400]
        max_tick = np.where(vlims[1] > max_ticks)[0]
        if len(max_tick) > 0:
            cb_max = max_ticks[
                max_tick[-1] + 1 if max_tick[-1] != len(max_ticks) - 1 else max_tick[-1]
            ]
        else:
            cb_max = max_ticks[0]

        cnorm = TwoSlopeNorm(
            vmin=cb_min,
            vcenter=0,
            vmax=cb_max,
        )

        # ----- #
        # Subplots.

        # Waveforms (kernels) illustrations.
        gs0w = gs0[ck, 0].subgridspec(3, 1)
        ax0w = fig0.add_subplot(gs0w[1])

        ax0w.plot(conv_kernel, c="k", linewidth=0.5)

        ax0w.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax0w.set_xticks([])
        ax0w.set_yticks([])

        gs1w = gs1[ck, 0].subgridspec(3, 1)
        ax1w = fig1.add_subplot(gs1w[1])

        ax1w.plot(conv_kernel, c="k", linewidth=0.5)

        ax1w.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax1w.set_xticks([])
        ax1w.set_yticks([])

        if ck < 2:
            gs2w = gs2[ck + 3, 0].subgridspec(3, 1)
            ax2w = fig2.add_subplot(gs2w[1])

            ax2w.plot(conv_kernel, c="k", linewidth=0.5)

            ax2w.spines[["top", "bottom", "left", "right"]].set_visible(False)
            ax2w.set_xticks([])
            ax2w.set_yticks([])

        # Colorbars.
        gs0c = gs0[ck, 6].subgridspec(1, 1)
        ax0c = fig0.add_subplot(gs0c[0])
        gs1c = gs1[ck, 6].subgridspec(1, 1)
        ax1c = fig1.add_subplot(gs1c[0])
        if ck < 2:
            gs2c = gs2[ck + 3, 6].subgridspec(1, 1)
            ax2c = fig2.add_subplot(gs2c[0])

        # ----- #
        # Topomaps.
        vlims_csp = []

        # ----- #
        # Figure 0: convolution amplitude without CSP.
        for ti, time_id in enumerate(time_ids):
            # Time window.
            if power_patterns == "windowed":
                time_window = [time_id - window_samples, time_id + window_samples + 1]
            elif power_patterns == "averaged":
                if ti == 0:
                    time_window = [base_id, time_id]
                else:
                    time_window = [time_ids[0], time_id + 1]

            # Subplots selction.
            gs00 = gs0[ck, ti + 1].subgridspec(1, 2)
            ax00 = fig0.add_subplot(gs00[0])
            ax01 = fig0.add_subplot(gs00[1])

            # Power % change - Class 1.
            conv_kernel_data_1 = np.mean(
                conv_kernel_data[lab_1, :, time_window[0] : time_window[1]] ** 2,
                axis=(0, 2),
            )
            conv_kernel_data_1 = (conv_kernel_data_1 - conv_base_1) / conv_base_1 * 100

            # Power % change - Class 2.
            conv_kernel_data_2 = np.mean(
                conv_kernel_data[lab_2, :, time_window[0] : time_window[1]] ** 2,
                axis=(0, 2),
            )
            conv_kernel_data_2 = (conv_kernel_data_2 - conv_base_2) / conv_base_2 * 100

            if ti == 0:
                cb0 = plt.colorbar(
                    ScalarMappable(norm=cnorm, cmap="RdBu_r"),
                    cax=ax0c,
                    ticks=[cb_min, 0.0, cb_max],
                )
                cb0.set_label(label="Δ power (%)", fontsize=textsizes[1])
                cb0.ax.tick_params(labelsize=textsizes[2], width=0.5, length=0.5)
                cb0.ax.yaxis.offsetText.set_fontsize(textsizes[2])

            # Power.
            plot_topomap(
                conv_kernel_data_1,
                info,
                ch_type="eeg",
                sensors=True,
                names=None,
                axes=ax00,
                cnorm=cnorm,
                show=False,
            )

            plot_topomap(
                conv_kernel_data_2,
                info,
                ch_type="eeg",
                sensors=True,
                names=None,
                axes=ax01,
                cnorm=cnorm,
                show=False,
            )

            # FOR Figure 1:
            # CSP data for color normalization.
            csp.fit_transform(
                np.copy(conv_kernel_data[:, :, time_window[0] : time_window[1]]), labels
            )

            vlims_csp.append([np.min(csp.patterns_), np.max(csp.patterns_)])

            # Title reflects time point.
            if ck == 0:
                ax00.set_title(
                    (
                        "t = {} ± {} s".format(time_points[ti], window_length * 5)
                        if power_patterns == "windowed"
                        else "t = {} s".format(time_points[ti])
                    ),
                    fontsize=textsizes[1],
                    loc="left",
                    fontweight="bold",
                )

        # FOR Figure 1:
        # CSP data plotting.
        vlims_csp_k = np.max(np.abs(vlims_csp)) * scalings["eeg"]
        cnorm_csp = Normalize(
            vmin=-vlims_csp_k,
            vmax=vlims_csp_k,
        )

        # ----- #
        # Figure 1: convolution with CSP.
        for ti, time_id in enumerate(time_ids):
            # Time window.
            time_window = [time_id - window_samples, time_id + window_samples + 1]

            # Time window.
            if power_patterns == "windowed":
                time_window = [time_id - window_samples, time_id + window_samples + 1]
            elif power_patterns == "averaged":
                if ti == 0:
                    time_window = [base_id, time_id]
                else:
                    time_window = [time_ids[0], time_id + 1]

            # Subplots selection.
            gs10 = gs1[ck, ti + 1].subgridspec(1, 2)
            ax10 = fig1.add_subplot(gs10[0])
            ax11 = fig1.add_subplot(gs10[1])

            # FOR Figure 2:
            if ck < 2:
                gs20 = gs2[ck + 3, ti + 1].subgridspec(1, 2)
                ax20 = fig2.add_subplot(gs20[0])
                ax21 = fig2.add_subplot(gs20[1])

            csp.fit_transform(
                np.copy(conv_kernel_data[:, :, time_window[0] : time_window[1]]), labels
            )

            # Colorbar.
            if ti == 0:
                cb1 = plt.colorbar(
                    ScalarMappable(norm=cnorm_csp, cmap="RdBu_r"),
                    cax=ax1c,
                    ticks=[vlims_csp_k, 0.0, vlims_csp_k],
                )
                cb1.set_label(label="Patterns (a.u.)", fontsize=textsizes[1])
                cb1.ax.tick_params(labelsize=textsizes[2], width=0.5, length=0.5)
                cb1.ax.title.set_fontsize(textsizes[2])
                cb1.set_ticks(
                    [np.around(-vlims_csp_k, decimals=4), 0, np.around(vlims_csp_k, 4)]
                )

                # FOR Figure 2:
                if ck < 2:
                    cb2 = plt.colorbar(
                        ScalarMappable(norm=cnorm_csp, cmap="RdBu_r"),
                        cax=ax2c,
                        ticks=[vlims_csp_k, 0.0, vlims_csp_k],
                    )
                    cb2.set_label(label="Patterns (a.u.)", fontsize=textsizes[1])
                    cb2.ax.tick_params(labelsize=textsizes[2], width=0.5, length=0.5)
                    cb2.ax.title.set_fontsize(textsizes[2])
                    cb2.set_ticks(
                        [
                            np.around(-vlims_csp_k, decimals=4),
                            0,
                            np.around(vlims_csp_k, 4),
                        ]
                    )

            # CSP patterns.
            csp.plot_patterns(
                info,
                ch_type="eeg",
                axes=[ax10, ax11] if csp_order == "mutual_info" else [ax11, ax10],
                cnorm=cnorm_csp,
                colorbar=False,
                name_format="",
                cbar_fmt="%1.1e",
                show=False,
            )

            # FOR Figure 2:
            if ck < 2:
                csp.plot_patterns(
                    info,
                    ch_type="eeg",
                    axes=[ax20, ax21] if csp_order == "mutual_info" else [ax21, ax20],
                    cnorm=cnorm_csp,
                    colorbar=False,
                    name_format="",
                    cbar_fmt="%1.1e",
                    show=False,
                )

            # Title reflects time point.
            if ck == 0:
                ax10.set_title(
                    (
                        "t = {} ± {} s".format(time_points[ti], window_length * 5)
                        if power_patterns == "windowed"
                        else "t = {} s".format(time_points[ti])
                    ),
                    fontsize=textsizes[1],
                    loc="left",
                    fontweight="bold",
                )

            # X label reflects CSP pattern.
            if ck == len(conv_kernels) - 1 and csp_order == "mutual_info":
                ax10.set_xlabel(
                    "CSP0",
                    fontsize=textsizes[1],
                )
                ax11.set_xlabel(
                    "CSP1",
                    fontsize=textsizes[1],
                )

                # FOR Figure 2:
                if ck < 2:
                    ax20.set_xlabel(
                        "CSP0",
                        fontsize=textsizes[1],
                    )
                    ax21.set_xlabel(
                        "CSP1",
                        fontsize=textsizes[1],
                    )

    # ----- #
    fe_vlims = []
    for filtered_epoch in all_epochs:

        for ti, time_id in enumerate(time_ids):
            # Time window.
            time_window = [time_id - window_samples, time_id + window_samples + 1]

            # Time window.
            if power_patterns == "windowed":
                time_window = [time_id - window_samples, time_id + window_samples + 1]
            elif power_patterns == "averaged":
                if ti == 0:
                    time_window = [base_id, time_id]
                else:
                    time_window = [time_ids[0], time_id + 1]

            csp.fit_transform(
                np.copy(filtered_epoch[:, :, time_window[0] : time_window[1]]), labels
            )

            fe_vlims.append([np.min(csp.patterns_), np.max(csp.patterns_)])

    vlims_csp_fe = np.max(np.abs(fe_vlims)) * scalings["eeg"]
    cnorm_csp_fe = Normalize(
        vmin=-vlims_csp_fe,
        vmax=vlims_csp_fe,
    )

    # Figure 2: filtering with CSP.
    for fe, filtered_epoch in enumerate(all_epochs):

        gs2w = gs2[fe, 0].subgridspec(3, 1)
        ax2w = fig2.add_subplot(gs2w[1])

        if fe == 0:
            ax2w.text(
                0.0,
                0.5,
                "Beta band\n{} - {} Hz".format(filter_bank[0][0], filter_bank[0][1]),
                fontsize=textsizes[1],
            )
        elif fe == 1:
            ax2w.text(
                0.0,
                0.5,
                "Mu band\n{} - {} Hz".format(filter_bank[1][0], filter_bank[1][1]),
                fontsize=textsizes[1],
            )
        elif fe == 2:
            ax2w.text(
                0.0,
                0.5,
                "Mu-Beta band\n{} - {} Hz".format(filter_bank[2][0], filter_bank[2][1]),
                fontsize=textsizes[1],
            )

        ax2w.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax2w.set_xticks([])
        ax2w.set_yticks([])

        for ti, time_id in enumerate(time_ids):

            # Time window.
            if power_patterns == "windowed":
                time_window = [time_id - window_samples, time_id + window_samples + 1]
            elif power_patterns == "averaged":
                if ti == 0:
                    time_window = [base_id, time_id]
                else:
                    time_window = [time_ids[0], time_id + 1]

            # Subplots selection.
            gs20 = gs2[fe, ti + 1].subgridspec(1, 2)
            ax20 = fig2.add_subplot(gs20[0])
            ax21 = fig2.add_subplot(gs20[1])

            # Colorbar.
            if ti == 0:

                gs2c = gs2[fe, 6].subgridspec(1, 1)
                ax2c = fig2.add_subplot(gs2c[0])

                cb2 = plt.colorbar(
                    ScalarMappable(norm=cnorm_csp_fe, cmap="RdBu_r"),
                    cax=ax2c,
                    ticks=[-vlims_csp_fe, 0.0, vlims_csp_fe],
                )
                cb2.set_label(label="Patterns (a.u.)", fontsize=textsizes[1])
                cb2.ax.tick_params(labelsize=textsizes[2], width=0.5, length=0.5)
                cb2.ax.title.set_fontsize(textsizes[2])
                cb2.set_ticks(
                    [
                        np.around(-vlims_csp_fe, decimals=4),
                        0,
                        np.around(vlims_csp_fe, 4),
                    ]
                )

            # Filters CSP patterns.
            csp.fit_transform(
                np.copy(filtered_epoch[:, :, time_window[0] : time_window[1]]), labels
            )

            csp.plot_patterns(
                info,
                ch_type="eeg",
                axes=[ax20, ax21] if csp_order == "mutual_info" else [ax21, ax20],
                cnorm=cnorm_csp_fe,
                colorbar=False,
                name_format="",
                cbar_fmt="%1.1e",
                show=False,
            )

            # Title reflects time point.
            if fe == 0:
                ax20.set_title(
                    (
                        "t = {} ± {} s".format(time_points[ti], window_length * 5)
                        if power_patterns == "windowed"
                        else "t = {} s".format(time_points[ti])
                    ),
                    fontsize=textsizes[1],
                    loc="left",
                    fontweight="bold",
                )

    # ----- #
    # Hands illustrations.
    for ti, _ in enumerate(time_ids):
        # Subplot selction.
        gs00 = gs0[6, ti + 1].subgridspec(1, 2)
        ax00 = fig0.add_subplot(gs00[0])
        ax01 = fig0.add_subplot(gs00[1])

        ax00.imshow(left_hand_img)
        ax01.imshow(right_hand_img)

        ax00.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax00.set_xticks([])
        ax00.set_yticks([])
        ax01.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax01.set_xticks([])
        ax01.set_yticks([])

        if csp_order == "alternate":
            # Subplot selction.
            gs10 = gs1[6, ti + 1].subgridspec(1, 2)
            ax10 = fig1.add_subplot(gs10[0])
            ax11 = fig1.add_subplot(gs10[1])

            ax10.imshow(left_hand_img)
            ax11.imshow(right_hand_img)

            ax10.spines[["top", "bottom", "left", "right"]].set_visible(False)
            ax10.set_xticks([])
            ax10.set_yticks([])
            ax11.spines[["top", "bottom", "left", "right"]].set_visible(False)
            ax11.set_xticks([])
            ax11.set_yticks([])

            gs20 = gs2[5, ti + 1].subgridspec(1, 2)
            ax20 = fig2.add_subplot(gs20[0])
            ax21 = fig2.add_subplot(gs20[1])

            ax20.imshow(left_hand_img)
            ax21.imshow(right_hand_img)

            ax20.spines[["top", "bottom", "left", "right"]].set_visible(False)
            ax20.set_xticks([])
            ax20.set_yticks([])
            ax21.spines[["top", "bottom", "left", "right"]].set_visible(False)
            ax21.set_xticks([])
            ax21.set_yticks([])

    # ----- #
    # SupTitles.
    fig0.suptitle(
        "{}, S{}".format(dataset_name, subject),
        fontsize=textsizes[0],
        fontweight="bold",
    )

    fig1.suptitle(
        "{}, S{}".format(dataset_name, subject),
        fontsize=textsizes[0],
        fontweight="bold",
    )

    fig2.suptitle(
        "{}, S{}".format(dataset_name, subject),
        fontsize=textsizes[0],
        fontweight="bold",
    )

    print("\n")

    # ----- #
    # Optional saving.
    if savefigs == True:
        fig0_name = "conv_ampl_spatial_patterns_{}.{}".format(
            power_patterns, plot_format
        )
        fig0.savefig(sub_dir + fig0_name, dpi=dpi, facecolor="w", edgecolor="w")
        fig1_name = "conv_csp_{}_spatial_patterns_{}.{}".format(
            csp_order_txt, power_patterns, plot_format
        )
        fig1.savefig(sub_dir + fig1_name, dpi=dpi, facecolor="w", edgecolor="w")
        fig2_name = "all_csp_{}_spatial_patterns_{}.{}".format(
            csp_order_txt, power_patterns, plot_format
        )
        fig2.savefig(sub_dir + fig2_name, dpi=dpi, facecolor="w", edgecolor="w")
    elif savefigs == False:
        plt.show()
