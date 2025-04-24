"""
Visualization of across-trials average burst probability modulation
per burst dictionary characteristic for a pair of channels.

Optionally perform cluster-based statistics and identify
statistically significant differences within- and across-
conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
from os.path import join, dirname

import mne
from mne.stats import permutation_cluster_test

from help_funcs import circumference, load_exp_variables
from time_res_features import compute_burst_rate_nd


def chars_modulation(
    subject,
    dataset,
    exp_variables,
    sub_dict_cond1,
    sub_dict_cond2,
    sub_dict_trials_cond1,
    sub_dict_trials_cond2,
    channels,
    sub_chars_dists,
    chars_groups,
    band_pass,
    binned_plot_time,
    bin_dt,
    task_time_lims,
    baseline_time_lims,
    tf_method,
    band,
    epochs,
    savepath,
    show_splits=False,
    show_stats=False,
    screen_res=[1920, 972],
    dpi=300,
    savefigs=True,
    plot_format="pdf",
):
    """
    Figures of the trial-averaged burst modulation across the time and characteristics axes for two
    channels (most interestingly C3 and C4) per experimental condition.

    Parameters
    ----------
    subject : int
              Integer indicating the subjects' data used for creating
              the burst dictionary.
    dataset: MOABB object
             Dataset from the MOABB for the analysis.
    exp_variables: dict
                   Dictionary containing some basic variables corresponding to 'dataset'.
    sub_dict_cond1: dict
                    Dictionary containing all detected bursts of 'subject' for condition 1.
    sub_dict_cond2 : dict
                    Dictionary containing all detected bursts of 'subject' for condition 2.
    sub_dict_trials_cond1: list
                           Indices of "condition 1" trials.
    sub_dict_trials_cond2: list
                           Indices of "condition 2" trials.
    channels: list
              Names of channels used while creation burst features. Search for and plot
              only C3 and C4.
    sub_chars_dists: list
                     List containing all burst scores for each burst dictionary
                     charecteristic (based on all trials) for 'subject'.
    chars_groups: int
                  Number of groups for the scores of burst characteristic axis
                  should be split into.
    band_pass: two-element list or numpy array
               Band pass limits for filtering the data while loading them.
    binned_plot_time: numpy array
                      Array representing the trimmed experimental time
                      with wide time steps, needed for an estimation
                      of burst rate in non-overlapping windows.
    bin_dt: float
            Time step for advancing to a new time window.
    task_time_lims: two-element list or numpy array
                    Start and end time of the task period (in seconds,
                    relative to the time-locked event).
    baseline_time_lims: two-element list or 1D array
                        Start and end time of the baseline period (in seconds,
                        relative to the time-locked event).
    tf_method: str {"wavelets", "superlets"}
               String indicating the algorithm used for burst extraction.
               Ignored if 'savefigs' is set to "False".
    band: str {"mu", "beta"}
          Select band for burst detection.
          Ignored if 'savefigs' is set to "False".
    epochs: MNE-python epochs object
            Epochs object corresponding to 'dataset'.
    savepath: str
              Parent directory that contains all results. Ignored if
              'savefigs' is set to "False".
    show_splits: bool, optional
                 If set to "True" show lines that indicate how the scores axis is
                 split in order to create features.
                 Defaults to "False".
    show_stats: bool, optional
                If set to "True" show contour lines that indicate statistically
                significant differences per channel/condition according to
                cluster-based permutation tests.
                Defaults to "False".
    screen_res: two-element list, optional
                Number of pixels for specifying the figure size in
                conjunction with 'dpi'.
                Defaults to [1920, 972].
    dpi: int, optional
         Number of dots per inch for specifying the figure size in
         conjunction with "screen_res".
         Defaults to 300.
    savefigs: bool, optional
              If set to "True" the visualizations are automatically
              saved. If set to "False" they are shown on screen.
              Defaults to "True".
    plot_format: str {"pdf", "png"}, optional
                File format. Prefer "pdf" for editing with vector graphics
                applications, or "png" for less space usage and better
                integration with presentations.
                Defaults to "pdf".
    """

    # ----- #
    # 1. Variables.
    # Number of available burst dictionary characteristics.
    n_chars = len(sub_chars_dists)

    # Nunber of subplots for waveforms.
    n_waves = 7

    # Average waveforms.
    cond1_rs = np.random.randint(
        0,
        len(sub_dict_cond1["subject"]),
        size=int(0.05 * len(sub_dict_cond1["subject"])),
    )
    cond2_rs = np.random.randint(
        0,
        len(sub_dict_cond2["subject"]),
        size=int(0.05 * len(sub_dict_cond2["subject"])),
    )
    cond1_ms = np.mean(sub_dict_cond1["waveform"][cond1_rs], axis=0)
    cond2_ms = np.mean(sub_dict_cond2["waveform"][cond2_rs], axis=0)
    mean_wave_shape = np.mean(np.vstack((cond1_ms, cond2_ms)), axis=0)
    wave_times = sub_dict_cond1["waveform_times"]

    # Count number of bursts in each time window and measure bin.
    trials_cond1 = np.unique(sub_dict_cond1["trial"])
    trials_cond2 = np.unique(sub_dict_cond2["trial"])

    # Trim time axis to exclude edge effects and define baseline.
    binned_plot_time = binned_plot_time[3:-3]
    baseline_bins = np.where(
        (binned_plot_time >= baseline_time_lims[0])
        & (binned_plot_time <= baseline_time_lims[1])
    )[0]

    # Binning of burst scores axis and smoothing kernel stds variables.
    score_bins = 41

    # ----- #
    # 2. Figure.
    # Figures initialization.
    if savefigs == False:
        fig = plt.figure(
            constrained_layout=False,
            figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
            dpi=dpi,
        )
        gs = fig.add_gridspec(
            nrows=1,
            ncols=3,
            width_ratios=[0.3, 1.05, 1.0],
            wspace=0.05,
            bottom=0.05,
            top=0.95,
            left=0.05,
            right=0.95,
        )
        gs10 = gs[0].subgridspec(
            nrows=(n_chars + 2) * n_waves, ncols=2, width_ratios=[0.3, 0.7]
        )
        title_size = 8
        label_size = 6
        tick_size = 6
        linew = [0.75, 0.75, 1.25, 1.5]
    else:
        fig = plt.figure(constrained_layout=False, figsize=(7, 4.5), dpi=dpi)
        gs = fig.add_gridspec(
            nrows=1, ncols=3, width_ratios=[0.30, 1.05, 1.0], wspace=0.15, left=0.05
        )
        gs10 = gs[0].subgridspec(
            nrows=(n_chars + 2) * n_waves,
            ncols=2,
            width_ratios=[0.30, 0.70],
            wspace=0.15,
        )
        title_size = 8
        label_size = 6
        tick_size = 6
        linew = [0.75, 0.75, 1.25, 1.5]
    gs11 = gs[1].subgridspec(
        nrows=n_chars + 2, ncols=4, width_ratios=[0.3, 1.5, 1.5, 1.5]
    )
    gs12 = gs[2].subgridspec(
        nrows=n_chars + 2, ncols=4, width_ratios=[1.5, 1.5, 1.5, 0.1]
    )

    # Figure name and rows' titles.
    fig.suptitle("Subject {}".format(subject), fontsize=title_size)
    ch_name = "C3_C4"
    channel_ids = [
        np.where(np.array(channels) == "C3")[0],
        np.where(np.array(channels) == "C4")[0],
    ]
    if channel_ids[0].size == 0 and channel_ids[1].size == 0:
        # Catch the case where the naming conventions differ.
        experimental_vars = load_exp_variables(
            json_filename=join(savepath, "variables.json")
        )
        _channels = experimental_vars["_channels"]
        channel_ids = [
            np.where(np.array(_channels) == "C3")[0],
            np.where(np.array(_channels) == "C4")[0],
        ]

    # Static hand icons.
    ax00 = fig.add_subplot(gs11[0, 1])
    ax01 = fig.add_subplot(gs11[0, 2])
    ax02 = fig.add_subplot(gs12[0, 1])
    ax03 = fig.add_subplot(gs12[0, 2])

    left_hand_img = mpimg.imread(join(dirname(__file__), "./left_hand.png"))
    right_hand_img = mpimg.imread(join(dirname(__file__), "./right_hand.png"))

    # np.unique(labels) results in an ordered array, therefore "cond1" always corresponds
    # to "left hand", and "cond2" to "right hand".
    titles = ["Left Hand", "Right Hand"]
    ax01.imshow(left_hand_img)
    ax02.imshow(right_hand_img)

    ax01.set_title(titles[0], fontsize=title_size, loc="center")
    ax02.set_title(titles[1], fontsize=title_size, loc="center")

    for ax in (ax00, ax01, ax02, ax03):
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Sensor locations.
    ax111 = fig.add_subplot(gs11[1, 1])
    ax112 = fig.add_subplot(gs11[1, 2])
    ax113 = fig.add_subplot(gs11[1, 3])

    ax120 = fig.add_subplot(gs12[1, 0])
    ax121 = fig.add_subplot(gs12[1, 1])
    ax122 = fig.add_subplot(gs12[1, 2])

    epochs.plot_sensors(ch_groups=np.array([channel_ids[0]]), axes=ax111)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[0]]), axes=ax120)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[1]]), axes=ax112)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[1]]), axes=ax121)
    epochs.plot_sensors(ch_groups=np.array([channel_ids]), axes=ax113)
    epochs.plot_sensors(ch_groups=np.array([channel_ids]), axes=ax122)

    # Iteratation over burst dictionary characteristics.
    chars = ["volume", "peak_amp_iter", "peak_freq", "fwhm_time", "fwhm_freq"]
    chars_titles = ["Vol.", "Amp.", "Peak fr.", "FWHM dur.", "FWHM fr."]
    for char_id, (_, char_dist, char_title) in enumerate(
        zip(chars, sub_chars_dists, chars_titles)
    ):

        # Common scores limits for all subjects.
        measure_bins = np.linspace(np.min(char_dist), np.max(char_dist), score_bins)
        scores_cond1 = char_dist[np.hstack(sub_dict_trials_cond1)]
        scores_cond2 = char_dist[np.hstack(sub_dict_trials_cond2)]

        # Limits in the metrics that are used to split each image in features.
        if chars_groups == 1:
            raise ValueError(
                "You need to specify at least 2 groups when creating features!"
            )
        else:
            iqrs = np.linspace(np.min(char_dist), np.max(char_dist), chars_groups + 1)
            scores_lims = []
            for i in range(chars_groups):
                scores_lims.append([iqrs[i], iqrs[i + 1]])
            scores_lims = np.array(scores_lims)

        # Subject burst rates.
        mv_burst_rate_cond1_1, mv_burst_rate_cond1_2, _ = compute_burst_rate_nd(
            sub_dict_cond1,
            trials_cond1,
            channel_ids,
            scores_cond1,
            measure_bins,
            binned_plot_time,
            bin_dt,
        )
        mv_burst_rate_cond2_1, mv_burst_rate_cond2_2, _ = compute_burst_rate_nd(
            sub_dict_cond2,
            trials_cond2,
            channel_ids,
            scores_cond2,
            measure_bins,
            binned_plot_time,
            bin_dt,
        )

        # Cluster based permutation tests for identification of regions of statistically
        # significant differences.
        if show_stats == True:
            F_obs_1, clusters_1, cluster_p_values_1, _ = permutation_cluster_test(
                [mv_burst_rate_cond1_1, mv_burst_rate_cond1_2],
                out_type="mask",
                tail=0,
                threshold=0.05,
            )
            F_obs_2, clusters_2, cluster_p_values_2, _ = permutation_cluster_test(
                [mv_burst_rate_cond2_1, mv_burst_rate_cond2_2],
                out_type="mask",
                tail=0,
                threshold=0.05,
            )
            F_obs_3, clusters_3, cluster_p_values_3, _ = permutation_cluster_test(
                [
                    mv_burst_rate_cond1_1 - mv_burst_rate_cond1_2,
                    mv_burst_rate_cond2_1 - mv_burst_rate_cond2_2,
                ],
                out_type="mask",
                tail=0,
                threshold=0.05,
            )
            F_obs_1 = F_obs_1 * np.nan
            F_obs_2 = F_obs_2 * np.nan
            F_obs_3 = F_obs_2 * np.nan

            for c, p_val in zip(clusters_1, cluster_p_values_1):
                if p_val <= 0.05:
                    F_obs_1[c] = 1
            for c, p_val in zip(clusters_2, cluster_p_values_2):
                if p_val <= 0.05:
                    F_obs_2[c] = 1
            for c, p_val in zip(clusters_3, cluster_p_values_3):
                if p_val <= 0.05:
                    F_obs_3[c] = 1

            F_obs_plot_1 = circumference(F_obs_1)
            F_obs_plot_2 = circumference(F_obs_2)
            F_obs_plot_3 = circumference(F_obs_3)

        # Baseline correction for visualization purposes.
        mhc_11 = np.mean(
            mv_burst_rate_cond1_1[:, baseline_bins, :], axis=(0, 1)
        ).reshape(1, -1)
        mhc_12 = np.mean(
            mv_burst_rate_cond1_2[:, baseline_bins, :], axis=(0, 1)
        ).reshape(1, -1)
        mhc_21 = np.mean(
            mv_burst_rate_cond2_1[:, baseline_bins, :], axis=(0, 1)
        ).reshape(1, -1)
        mhc_22 = np.mean(
            mv_burst_rate_cond2_2[:, baseline_bins, :], axis=(0, 1)
        ).reshape(1, -1)

        mv_burst_rate_cond1_1 = (mv_burst_rate_cond1_1 - mhc_11) / mhc_11 * 100
        mv_burst_rate_cond1_2 = (mv_burst_rate_cond1_2 - mhc_12) / mhc_12 * 100
        mv_burst_rate_cond2_1 = (mv_burst_rate_cond2_1 - mhc_21) / mhc_21 * 100
        mv_burst_rate_cond2_2 = (mv_burst_rate_cond2_2 - mhc_22) / mhc_22 * 100

        # Average over trials for visualization.
        mh1_1 = np.mean(mv_burst_rate_cond1_1, axis=0)
        mh1_2 = np.mean(mv_burst_rate_cond1_2, axis=0)
        mh2_1 = np.mean(mv_burst_rate_cond2_1, axis=0)
        mh2_2 = np.mean(mv_burst_rate_cond2_2, axis=0)

        # Plots intialization.
        ax100_2 = fig.add_subplot(gs10[3 + n_waves * (char_id + 2), 0])

        ax101_0 = fig.add_subplot(gs10[1 + n_waves * (char_id + 2), 1])
        ax101_1 = fig.add_subplot(gs10[2 + n_waves * (char_id + 2), 1])
        ax101_2 = fig.add_subplot(gs10[3 + n_waves * (char_id + 2), 1])
        ax101_3 = fig.add_subplot(gs10[4 + n_waves * (char_id + 2), 1])
        ax101_4 = fig.add_subplot(gs10[5 + n_waves * (char_id + 2), 1])

        ax110 = fig.add_subplot(gs11[char_id + 2, 0])
        ax111 = fig.add_subplot(gs11[char_id + 2, 1])
        ax112 = fig.add_subplot(gs11[char_id + 2, 2])
        ax113 = fig.add_subplot(gs11[char_id + 2, 3])

        ax120 = fig.add_subplot(gs12[char_id + 2, 0])
        ax121 = fig.add_subplot(gs12[char_id + 2, 1])
        ax122 = fig.add_subplot(gs12[char_id + 2, 2])
        ax123 = fig.add_subplot(gs12[char_id + 2, 3])

        # Axis 100: NULL.

        # Axis 10: Burst waveform shape.
        wave_colors = plt.cm.cool(np.linspace(0, 1, 5))

        ext_neg_1 = np.where(scores_cond1 <= scores_lims[0][1])[0]
        mid_neg_1 = np.where(
            (scores_cond1 > scores_lims[1][0]) & (scores_cond1 <= scores_lims[1][1])
        )[0]
        ext_pos_1 = np.where(scores_cond1 >= scores_lims[-1][0])[0]
        mid_pos_1 = np.where(
            (scores_cond1 > scores_lims[-2][0]) & (scores_cond1 <= scores_lims[-1][0])
        )[0]

        ext_neg_2 = np.where(scores_cond2 <= scores_lims[0][1])[0]
        mid_neg_2 = np.where(
            (scores_cond2 > scores_lims[1][0]) & (scores_cond2 <= scores_lims[1][1])
        )[0]
        ext_pos_2 = np.where(scores_cond2 >= scores_lims[-1][0])[0]
        mid_pos_2 = np.where(
            (scores_cond2 > scores_lims[-2][0]) & (scores_cond2 <= scores_lims[-1][0])
        )[0]

        wave_neg = np.mean(
            np.vstack(
                (
                    sub_dict_cond1["waveform"][ext_neg_1],
                    sub_dict_cond2["waveform"][ext_neg_2],
                )
            ),
            axis=0,
        )
        wave_mneg = np.mean(
            np.vstack(
                (
                    sub_dict_cond1["waveform"][mid_neg_1],
                    sub_dict_cond2["waveform"][mid_neg_2],
                )
            ),
            axis=0,
        )
        wave_pos = np.mean(
            np.vstack(
                (
                    sub_dict_cond1["waveform"][ext_pos_1],
                    sub_dict_cond2["waveform"][ext_pos_2],
                )
            ),
            axis=0,
        )
        wave_mpos = np.mean(
            np.vstack(
                (
                    sub_dict_cond1["waveform"][mid_pos_1],
                    sub_dict_cond2["waveform"][mid_pos_2],
                )
            ),
            axis=0,
        )

        ax101_0.plot(wave_times, wave_pos, color=wave_colors[-1], linewidth=linew[0])
        ax101_1.plot(wave_times, wave_mpos, color=wave_colors[-2], linewidth=linew[0])
        ax101_2.plot(wave_times, mean_wave_shape, color="k", linewidth=linew[0])
        ax101_3.plot(wave_times, wave_mneg, color=wave_colors[1], linewidth=linew[0])
        ax101_4.plot(wave_times, wave_neg, color=wave_colors[0], linewidth=linew[0])

        for ax in (ax100_2, ax101_0, ax101_1, ax101_2, ax101_3, ax101_4):
            ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        ax100_2.set_ylabel(
            "{}".format(char_title), fontsize=label_size, fontweight="bold", rotation=0
        )

        # Axis 110: NULL.

        # Axes 111-3/120-2: bursts characteristics modulation.
        minima = np.nanmin((np.min(mh1_1), np.min(mh1_2), np.min(mh2_1), np.min(mh2_2)))
        maxima = np.nanmax((np.max(mh1_1), np.max(mh1_2), np.max(mh2_1), np.max(mh2_2)))
        divnorm = colors.TwoSlopeNorm(vmin=minima, vcenter=0, vmax=maxima)
        contournorm = colors.CenteredNorm(vcenter=0.5, halfrange=-0.1)

        im11 = ax111.imshow(
            mh1_1.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=divnorm,
            extent=(
                binned_plot_time[0],
                binned_plot_time[-1],
                measure_bins[0],
                measure_bins[-1],
            ),
        )
        ax112.imshow(
            mh1_2.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=divnorm,
            extent=(
                binned_plot_time[0],
                binned_plot_time[-1],
                measure_bins[0],
                measure_bins[-1],
            ),
        )
        ax113.imshow(
            mh1_1.T - mh1_2.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=divnorm,
            extent=(
                binned_plot_time[0],
                binned_plot_time[-1],
                measure_bins[0],
                measure_bins[-1],
            ),
        )

        if show_stats == True:
            ax111.imshow(
                F_obs_plot_1.T,
                aspect="auto",
                origin="lower",
                cmap="spring",
                norm=contournorm,
                interpolation="none",
                extent=(
                    binned_plot_time[0],
                    binned_plot_time[-1],
                    measure_bins[0],
                    measure_bins[-1],
                ),
            )
            ax112.imshow(
                F_obs_plot_1.T,
                aspect="auto",
                origin="lower",
                cmap="spring",
                norm=contournorm,
                interpolation="none",
                extent=(
                    binned_plot_time[0],
                    binned_plot_time[-1],
                    measure_bins[0],
                    measure_bins[-1],
                ),
            )
            ax113.imshow(
                F_obs_plot_3.T,
                aspect="auto",
                origin="lower",
                cmap="PiYG",
                norm=contournorm,
                interpolation="none",
                extent=(
                    binned_plot_time[0],
                    binned_plot_time[-1],
                    measure_bins[0],
                    measure_bins[-1],
                ),
            )

        ax120.imshow(
            mh2_1.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=divnorm,
            extent=(
                binned_plot_time[0],
                binned_plot_time[-1],
                measure_bins[0],
                measure_bins[-1],
            ),
        )
        ax121.imshow(
            mh2_2.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=divnorm,
            extent=(
                binned_plot_time[0],
                binned_plot_time[-1],
                measure_bins[0],
                measure_bins[-1],
            ),
        )
        ax122.imshow(
            mh2_1.T - mh2_2.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=divnorm,
            extent=(
                binned_plot_time[0],
                binned_plot_time[-1],
                measure_bins[0],
                measure_bins[-1],
            ),
        )

        if show_stats == True:
            ax120.imshow(
                F_obs_plot_2.T,
                aspect="auto",
                origin="lower",
                cmap="spring",
                norm=contournorm,
                interpolation="none",
                extent=(
                    binned_plot_time[0],
                    binned_plot_time[-1],
                    measure_bins[0],
                    measure_bins[-1],
                ),
            )
            ax121.imshow(
                F_obs_plot_2.T,
                aspect="auto",
                origin="lower",
                cmap="spring",
                norm=contournorm,
                interpolation="none",
                extent=(
                    binned_plot_time[0],
                    binned_plot_time[-1],
                    measure_bins[0],
                    measure_bins[-1],
                ),
            )
            ax122.imshow(
                F_obs_plot_3.T,
                aspect="auto",
                origin="lower",
                cmap="PiYG",
                norm=contournorm,
                interpolation="none",
                extent=(
                    binned_plot_time[0],
                    binned_plot_time[-1],
                    measure_bins[0],
                    measure_bins[-1],
                ),
            )

        # Task limits and feature splits.
        for ax in (ax111, ax112, ax113, ax120, ax121, ax122):
            ax.axvline(task_time_lims[0], linestyle=":", color="k", linewidth=linew[1])
            ax.axvline(task_time_lims[1], linestyle=":", color="k", linewidth=linew[1])
            if show_splits == True:
                for lim in scores_lims[:-1, 1]:
                    ax.axhline(
                        lim, linestyle="--", color="tab:gray", linewidth=linew[1]
                    )
            ax.tick_params(axis="both", labelsize=tick_size)

        # Axis 4: colorbar.
        min_ticks = [-5, -10, -25, -50, -75, -100, -150, -200, -300, -400]
        min_tick = np.where(minima < min_ticks)[0]
        cb_min = min_ticks[min_tick[-1]]

        max_ticks = [5, 10, 25, 50, 75, 100, 150, 200, 300, 400]
        max_tick = np.where(maxima > max_ticks)[0]
        cb_max = max_ticks[max_tick[-1]]

        cb = fig.colorbar(
            im11, cax=ax123, ticks=[cb_min, 0.0, cb_max], label="Burst rate\nchange (%)"
        )
        cb.set_label(label="Burst rate\nchange (%)", fontsize=label_size)
        cb.ax.tick_params(labelsize=label_size)

        # Labels, ticks and titles.
        if char_id == 0:
            # Titles.
            ax111.set_title("C3", fontsize=label_size, loc="left")
            ax112.set_title("C4", fontsize=label_size, loc="left")
            ax113.set_title("C3 - C4", fontsize=label_size, loc="left")
            ax120.set_title("C3", fontsize=label_size, loc="left")
            ax121.set_title("C4", fontsize=label_size, loc="left")
            ax122.set_title("C3 - C4", fontsize=label_size, loc="left")

        ax110.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax110.set_xticks([])
        ax110.set_yticks([])

        if char_id != n_chars - 1:
            for ax in (ax111, ax112, ax113, ax120, ax121, ax122):
                ax.set_xticks(task_time_lims)
                ax.set_xticklabels([])
        else:
            for ax in (ax111, ax112, ax113, ax120, ax121, ax122):
                ax.set_xticks(task_time_lims)
                ax.set_xticklabels(task_time_lims, fontsize=label_size)
        if char_id == n_chars - 1:
            for ax in (ax111, ax112, ax113, ax120, ax121, ax122):
                ax.set_xlabel("Time (s)", fontsize=label_size)

        for ax in (ax112, ax113, ax120, ax121, ax122):
            ax.set_yticklabels([])
        ax111.get_yaxis().get_offset_text().set_fontsize(label_size)
        ax111.get_yaxis().get_offset_text().set_position((0.0, 0.5))

    # Optional saving.
    if savefigs == True:
        savepath = join(savepath, "sub_{}/".format(subject))
        fig_name = join(
            savepath,
            "{}_burst_chars_features_modulation_{}_{}.{}".format(
                band, tf_method, ch_name, plot_format
            ),
        )
        fig.savefig(fig_name, dpi=dpi, facecolor="w", edgecolor="w")
        plt.close("all")
    elif savefigs == False:
        plt.show()
