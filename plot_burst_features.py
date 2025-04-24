"""
Visualization of across-trials average burst probability modulation
per component axis for a pair of channels.

Optionally perform cluster-based statistics and identify
statistically significant differences within- and across-
conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.image as mpimg
import imageio
import warnings
from os import makedirs
from os.path import join, dirname, exists

import mne
from mne.stats import permutation_cluster_test

from help_funcs import circumference, load_exp_variables, confidence_ellipse
from time_res_features import compute_burst_rate, compute_burst_rate_nd


def features_modulation(
    subject,
    exp_variables,
    sub_dict_cond1,
    sub_dict_cond2,
    sub_dict_trials_cond1,
    sub_dict_trials_cond2,
    channels,
    sub_scores_dists,
    comps_to_analyze,
    comps_to_vis,
    comps_groups,
    binned_plot_time,
    bin_dt,
    task_time_lims,
    baseline_time_lims,
    tf_method,
    band,
    solver,
    epochs,
    epochs_power_dict,
    savepath,
    baseline_correction="independent",
    show_splits=False,
    show_stats=False,
    show_sample=False,
    screen_res=[1920, 972],
    dpi=300,
    savefigs=True,
    plot_format="pdf",
):
    """
    Figures of the trial-averaged burst modulation across the time and scores axes for two
    channels (most interestingly C3 and C4) per component and experimental condition.

    Parameters
    ----------
    subject : int
              Integer indicating the subjects' data used for creating
              the burst dictionary.
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
    sub_scores_dists: list
                      List containing all burst scores along each principal
                      component axis (based on all trials) for 'subject'.
    comps_to_analyze: list or numpy array
                      List of the indices of components used for the visualization.
    comps_to_vis: list or numpy array
                  List of the indices of components used for the supplementary
                  visualization.
    comps_groups: int
                  Number of groups the scores of each component axis should
                  be split into.
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
    solver: str, {"pca", "csp"}
            Dimensionality reduction algorithm. Ignored if 'savefigs' is set to
            "False".
    epochs: MNE-python epochs object
            Epochs object corresponding to 'dataset'.
    epochs_power_dict: dict
                       Dictionary of arrays containing the trial-averaged,
                       baseline-corrected Hilbert envelope power for a given
                       condition, and corresponding sem.
    savepath: str
              Parent directory that contains all results. Ignored if
              'savefigs' is set to "False".
    baseline_correction: str {"independent", "channel", "condition"}, optional
                         String indicating whether to apply baseline correction
                         per channel and condition, across conditions for each
                         channel or across channels for each condition.
                         Defaults to "independent".
    show_splits: bool, optional
                 If set to "True" show lines that indicate how the scores axis is
                 split in order to create features.
                 Defaults to "False".
    show_stats: bool, optional
                If set to "True" show contour lines that indicate statistically
                significant differences per channel/condition according to
                cluster-based permutation tests.
                Defaults to "False".
    show_sample: bool, optional
                 If set to "True" produce a child figure with only a single plot
                 for pedagogic purposes.
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
    # Number of available components.
    n_comps = len(comps_to_analyze)

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

    # Identify trials corresponding to each condition.
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
    # 2. Figures.
    # Figures initialization.
    fig = plt.figure(constrained_layout=False, figsize=(7, 4.5), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=1, ncols=3, width_ratios=[0.20, 1.05, 1.0], wspace=0.15, left=0.05
    )
    gs10 = gs[0].subgridspec(
        nrows=(n_comps + 2) * n_waves, ncols=2, width_ratios=[0.10, 0.90], wspace=0.15
    )

    fig_2 = plt.figure(constrained_layout=False, figsize=(7, 5), dpi=dpi)
    gs_2 = fig_2.add_gridspec(
        nrows=5,
        ncols=2,
        hspace=0.50,
        wspace=0.25,
        bottom=0.10,
        top=0.90,
        left=0.05,
        right=0.95,
        height_ratios=[1.5, 1.5, 2, 2, 2],
        width_ratios=[0.3, 6],
    )

    title_size = 8
    label_size = 6
    tick_size = 6
    linew = [0.75, 0.75, 1.25, 1.5]

    gs11 = gs[1].subgridspec(
        nrows=n_comps + 2, ncols=4, width_ratios=[0.3, 1.5, 1.5, 1.5]
    )
    gs12 = gs[2].subgridspec(
        nrows=n_comps + 2, ncols=4, width_ratios=[1.5, 1.5, 1.5, 0.1]
    )

    gs_2_w = gs_2[:, 0].subgridspec(nrows=5 * n_waves, ncols=1)
    gs_2_0 = gs_2[0, 1].subgridspec(nrows=1, ncols=6, width_ratios=[1, 1, 1, 1, 1, 1])
    gs_2_1 = gs_2[1, 1].subgridspec(nrows=1, ncols=6, width_ratios=[1, 1, 1, 1, 1, 1])
    gs_2_2 = gs_2[2, 1].subgridspec(nrows=1, ncols=6, width_ratios=[1, 1, 1, 1, 1, 1])
    gs_2_3 = gs_2[3, 1].subgridspec(nrows=1, ncols=6, width_ratios=[1, 1, 1, 1, 1, 1])
    gs_2_4 = gs_2[4, 1].subgridspec(nrows=1, ncols=6, width_ratios=[1, 1, 1, 1, 1, 1])

    # Figure 2 variables.
    ax_2_colors = ["c", "k", "m"]
    zorders = [2, 1, 3]
    n_collapsed_groups = 5

    # Y limits initialization.
    modulation_scores_min = []
    modulation_scores_max = []
    bag_of_axes = []

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
        channel_vals = [
            channels[int(np.where(np.array(_channels) == "C3")[0])],
            channels[int(np.where(np.array(_channels) == "C4")[0])],
        ]
        channels = np.sort(np.array(channels).astype(np.int)).astype(np.str).tolist()
        channel_ids = [
            np.where(np.array(channels) == channel_vals[0])[0],
            np.where(np.array(channels) == channel_vals[1])[0],
        ]

    # Optionally produce a sample visualization.
    if show_sample == True:
        fig_sample = plt.figure(figsize=(3.0, 2.0))
        gs_sample = fig_sample.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[0.15, 0.85],
            wspace=0.25,
            bottom=0.15,
            top=0.85,
            left=0.05,
            right=0.85,
        )
        gs_sample_00 = gs_sample[0].subgridspec(nrows=n_waves - 2, ncols=1)
        gs_sample_01 = gs_sample[1].subgridspec(
            nrows=1, ncols=2, width_ratios=[0.95, 0.05], wspace=0.05
        )
        gs_sample_03 = gs_sample[1].subgridspec(
            nrows=comps_groups, ncols=2, width_ratios=[0.95, 0.05], wspace=0.05
        )
        ax_sample_01 = fig_sample.add_subplot(gs_sample_01[0])
        ax_sample_02 = fig_sample.add_subplot(gs_sample_01[1])

    # Static hand icons.
    ax00 = fig.add_subplot(gs11[0, 1])
    ax01 = fig.add_subplot(gs11[0, 2])
    ax02 = fig.add_subplot(gs12[0, 1])
    ax03 = fig.add_subplot(gs12[0, 2])
    ax_2_00 = fig_2.add_subplot(gs_2_0[1])
    ax_2_01 = fig_2.add_subplot(gs_2_0[4])

    left_hand_img = mpimg.imread(join(dirname(__file__), "./left_hand.png"))
    right_hand_img = mpimg.imread(join(dirname(__file__), "./right_hand.png"))

    # np.unique(labels) results in an ordered array, therefore "cond1" always corresponds
    # to "left hand", and "cond2" to "right hand".
    titles = ["Left Hand", "Right Hand"]
    ax01.imshow(left_hand_img)
    ax02.imshow(right_hand_img)
    ax_2_00.imshow(left_hand_img)
    ax_2_01.imshow(right_hand_img)

    ax01.set_title(titles[0], fontsize=title_size, loc="center")
    ax02.set_title(titles[1], fontsize=title_size, loc="center")
    ax_2_00.set_title(titles[0], fontsize=title_size, loc="center")
    ax_2_01.set_title(titles[1], fontsize=title_size, loc="center")

    for ax in (ax00, ax01, ax02, ax03, ax_2_00, ax_2_01):
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # ----- #
    # 3. Plotting.
    # Sensor locations.
    ax111 = fig.add_subplot(gs11[1, 1])
    ax112 = fig.add_subplot(gs11[1, 2])
    ax113 = fig.add_subplot(gs11[1, 3])
    ax120 = fig.add_subplot(gs12[1, 0])
    ax121 = fig.add_subplot(gs12[1, 1])
    ax122 = fig.add_subplot(gs12[1, 2])

    ax_2_10 = fig_2.add_subplot(gs_2_1[0])
    ax_2_11 = fig_2.add_subplot(gs_2_1[1])
    ax_2_12 = fig_2.add_subplot(gs_2_1[2])
    ax_2_13 = fig_2.add_subplot(gs_2_1[3])
    ax_2_14 = fig_2.add_subplot(gs_2_1[4])
    ax_2_15 = fig_2.add_subplot(gs_2_1[5])

    epochs.plot_sensors(ch_groups=np.array([channel_ids[0]]), axes=ax111)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[0]]), axes=ax120)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[1]]), axes=ax112)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[1]]), axes=ax121)
    epochs.plot_sensors(ch_groups=np.array([channel_ids]), axes=ax113)
    epochs.plot_sensors(ch_groups=np.array([channel_ids]), axes=ax122)

    epochs.plot_sensors(ch_groups=np.array([channel_ids[0]]), axes=ax_2_10)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[0]]), axes=ax_2_13)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[1]]), axes=ax_2_11)
    epochs.plot_sensors(ch_groups=np.array([channel_ids[1]]), axes=ax_2_14)
    epochs.plot_sensors(ch_groups=np.array([channel_ids]), axes=ax_2_12)
    epochs.plot_sensors(ch_groups=np.array([channel_ids]), axes=ax_2_15)

    # Titles.
    ax111.set_title("C3", fontsize=label_size, loc="left")
    ax112.set_title("C4", fontsize=label_size, loc="left")
    ax113.set_title("C3 - C4", fontsize=label_size, loc="left")
    ax120.set_title("C3", fontsize=label_size, loc="left")
    ax121.set_title("C4", fontsize=label_size, loc="left")
    ax122.set_title("C3 - C4", fontsize=label_size, loc="left")

    ax_2_10.set_title("C3", fontsize=label_size, loc="left")
    ax_2_11.set_title("C4", fontsize=label_size, loc="left")
    ax_2_12.set_title("C3 - C4", fontsize=label_size, loc="left")
    ax_2_13.set_title("C3", fontsize=label_size, loc="left")
    ax_2_14.set_title("C4", fontsize=label_size, loc="left")
    ax_2_15.set_title("C3 - C4", fontsize=label_size, loc="left")

    # Iteratation over components.
    for comp, (comp_name, scd) in enumerate(zip(comps_to_analyze, sub_scores_dists)):
        # Common scores limits for all subjects.
        measure_bins = np.linspace(np.min(scd), np.max(scd), score_bins)
        scores_cond1 = scd[np.hstack(sub_dict_trials_cond1)]
        scores_cond2 = scd[np.hstack(sub_dict_trials_cond2)]

        # Limits in the metrics that are used to split each image in features.
        if comps_groups == 1:
            raise ValueError(
                "You need to specify at least 2 groups when creating burst features!"
            )
        else:
            # Common score binning across conditions.
            iqrs = np.linspace(np.min(scd), np.max(scd), comps_groups + 1)
            scores_lims = []
            for i in range(comps_groups):
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
        if baseline_correction == "independent":
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

        elif baseline_correction == "channel":
            mhc_121 = np.mean(
                np.vstack(
                    (
                        mv_burst_rate_cond1_1[:, baseline_bins, :],
                        mv_burst_rate_cond2_1[:, baseline_bins, :],
                    )
                ),
                axis=(0, 1),
            ).reshape(1, -1)
            mhc_122 = np.mean(
                np.vstack(
                    (
                        mv_burst_rate_cond1_2[:, baseline_bins, :],
                        mv_burst_rate_cond2_2[:, baseline_bins, :],
                    )
                ),
                axis=(0, 1),
            ).reshape(1, -1)

            mv_burst_rate_cond1_1 = (mv_burst_rate_cond1_1 - mhc_121) / mhc_121 * 100
            mv_burst_rate_cond1_2 = (mv_burst_rate_cond1_2 - mhc_122) / mhc_122 * 100
            mv_burst_rate_cond2_1 = (mv_burst_rate_cond2_1 - mhc_121) / mhc_121 * 100
            mv_burst_rate_cond2_2 = (mv_burst_rate_cond2_2 - mhc_122) / mhc_122 * 100

        elif baseline_correction == "condition":
            mhc_112 = np.mean(
                np.vstack(
                    (
                        mv_burst_rate_cond1_1[:, baseline_bins, :],
                        mv_burst_rate_cond1_2[:, baseline_bins, :],
                    )
                ),
                axis=(0, 1),
            ).reshape(1, -1)
            mhc_212 = np.mean(
                np.vstack(
                    (
                        mv_burst_rate_cond2_1[:, baseline_bins, :],
                        mv_burst_rate_cond2_2[:, baseline_bins, :],
                    )
                ),
                axis=(0, 1),
            ).reshape(1, -1)

            mv_burst_rate_cond1_1 = (mv_burst_rate_cond1_1 - mhc_112) / mhc_112 * 100
            mv_burst_rate_cond1_2 = (mv_burst_rate_cond1_2 - mhc_112) / mhc_112 * 100
            mv_burst_rate_cond2_1 = (mv_burst_rate_cond2_1 - mhc_212) / mhc_212 * 100
            mv_burst_rate_cond2_2 = (mv_burst_rate_cond2_2 - mhc_212) / mhc_212 * 100

        # Average over trials for visualization.
        mh1_1 = np.mean(mv_burst_rate_cond1_1, axis=0)
        mh1_2 = np.mean(mv_burst_rate_cond1_2, axis=0)
        mh2_1 = np.mean(mv_burst_rate_cond2_1, axis=0)
        mh2_2 = np.mean(mv_burst_rate_cond2_2, axis=0)

        # Subplots initialization.
        ax100_2 = fig.add_subplot(gs10[3 + n_waves * (comp + 2), 0])

        ax101_0 = fig.add_subplot(gs10[1 + n_waves * (comp + 2), 1])
        ax101_1 = fig.add_subplot(gs10[2 + n_waves * (comp + 2), 1])
        ax101_2 = fig.add_subplot(gs10[3 + n_waves * (comp + 2), 1])
        ax101_3 = fig.add_subplot(gs10[4 + n_waves * (comp + 2), 1])
        ax101_4 = fig.add_subplot(gs10[5 + n_waves * (comp + 2), 1])

        ax110 = fig.add_subplot(gs11[comp + 2, 0])
        ax111 = fig.add_subplot(gs11[comp + 2, 1])
        ax112 = fig.add_subplot(gs11[comp + 2, 2])
        ax113 = fig.add_subplot(gs11[comp + 2, 3])

        ax120 = fig.add_subplot(gs12[comp + 2, 0])
        ax121 = fig.add_subplot(gs12[comp + 2, 1])
        ax122 = fig.add_subplot(gs12[comp + 2, 2])
        ax123 = fig.add_subplot(gs12[comp + 2, 3])

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

        if solver == "pca":
            comp_title = "PC"
        elif solver == "csp":
            comp_title = "F"
        ax100_2.set_ylabel(
            "{} {}".format(comp_title, comp_name),
            fontsize=label_size,
            fontweight="bold",
            rotation=0,
        )

        # Axis 110: NULL.

        # Axes 111-3/120-2: bursts features modulation.
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

        # ----- #
        # Sample figure.
        if show_sample == True and comp_name == comps_to_vis[0]:
            warnings.warn(
                "Sample figure only plots a single condition of the 4th component.",
                UserWarning(),
            )

            # Burst waveform shape.
            ax_sample_00_0 = fig_sample.add_subplot(gs_sample_00[0])
            ax_sample_00_1 = fig_sample.add_subplot(gs_sample_00[1])
            ax_sample_00_2 = fig_sample.add_subplot(gs_sample_00[2])
            ax_sample_00_3 = fig_sample.add_subplot(gs_sample_00[3])
            ax_sample_00_4 = fig_sample.add_subplot(gs_sample_00[4])

            ax_sample_00_0.plot(
                wave_times, wave_pos, color=wave_colors[-1], linewidth=linew[2]
            )
            ax_sample_00_1.plot(
                wave_times, wave_mpos, color=wave_colors[-2], linewidth=linew[2]
            )
            ax_sample_00_2.plot(
                wave_times, mean_wave_shape, color="k", linewidth=linew[2]
            )
            ax_sample_00_3.plot(
                wave_times, wave_mneg, color=wave_colors[1], linewidth=linew[2]
            )
            ax_sample_00_4.plot(
                wave_times, wave_neg, color=wave_colors[0], linewidth=linew[2]
            )

            for ax in (
                ax_sample_00_0,
                ax_sample_00_1,
                ax_sample_00_2,
                ax_sample_00_3,
                ax_sample_00_4,
            ):
                ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])

            # Burst features modulation.
            im_sample = ax_sample_01.imshow(
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

            ax_sample_01.axvline(
                task_time_lims[0], linestyle=":", color="k", linewidth=linew[2]
            )
            ax_sample_01.axvline(
                task_time_lims[1], linestyle=":", color="k", linewidth=linew[2]
            )

            for lim in scores_lims[:-1, 1]:
                ax_sample_01.axhline(
                    lim, linestyle="--", color="tab:gray", linewidth=linew[2]
                )

            # Average modulation across groups.
            sb = mh2_1.T.shape[0] // comps_groups
            for cg in range(comps_groups):
                ax_sample_03 = fig_sample.add_subplot(
                    gs_sample_03[comps_groups - (cg + 1), 0]
                )

                start = cg * sb
                end = start + sb
                if cg != comps_groups - 1:
                    split = mh2_1.T[start:end, :]
                else:
                    split = mh2_1.T[start:, :]

                modulation = np.mean(split, axis=0)

                ax_sample_03.plot(
                    binned_plot_time[:-1], modulation, color="k", linewidth=linew[3]
                )

                ax_sample_03.set_ylim([np.min(mh2_1.T), np.max(mh2_1.T)])

                ax_sample_03.margins(x=0)
                ax_sample_03.patch.set_alpha(0.0)
                ax_sample_03.spines[["top", "bottom", "left", "right"]].set_visible(
                    False
                )
                ax_sample_03.set_xticks([])
                ax_sample_03.set_yticks([])
                ax_sample_03.set_xticklabels([])
                ax_sample_03.set_yticklabels([])

            # Colorbar, title, labels.
            cbs = fig_sample.colorbar(im_sample, cax=ax_sample_02)
            cbs.set_label(label="Burst rate\nchange (%)", fontsize=label_size)
            cbs.ax.tick_params(labelsize=label_size)

            fig_sample.suptitle("Subject {}".format(subject), fontsize=title_size)
            ax_sample_01.set_xlabel("Time (s)", fontsize=label_size)
            ax_sample_01.set_ylabel("Score (a.u.)", fontsize=label_size)
            ax_sample_01.tick_params(axis="both", labelsize=tick_size)
        # ----- #

        # ----- #
        # Collapsed features figure.

        # Only plot if splitting into five groups and inluding specific components.
        if comp_name == comps_to_vis[0]:
            warnings.warn(
                "Groupped subplots will only be plotted if visualizing 5 splits.",
                UserWarning,
            )

        if comp_name == comps_to_vis[0] or comp_name == comps_to_vis[1]:
            # Axes initialization.
            if comp_name == comps_to_vis[0]:
                row_id = 3
            elif comp_name == comps_to_vis[1]:
                row_id = 4

            ax_2_w0 = fig_2.add_subplot(gs_2_w[2 + n_waves * row_id, 0])
            ax_2_w1 = fig_2.add_subplot(gs_2_w[3 + n_waves * row_id, 0])
            ax_2_w2 = fig_2.add_subplot(gs_2_w[4 + n_waves * row_id, 0])

            # Select subgrid.
            proper_gs = gs_2_3 if (comp_name == comps_to_vis[0]) else gs_2_4

            ax_2_p0 = fig_2.add_subplot(proper_gs[0])
            ax_2_p1 = fig_2.add_subplot(proper_gs[1])
            ax_2_p2 = fig_2.add_subplot(proper_gs[2])
            ax_2_p3 = fig_2.add_subplot(proper_gs[3])
            ax_2_p4 = fig_2.add_subplot(proper_gs[4])
            ax_2_p5 = fig_2.add_subplot(proper_gs[5])

            ax_2_p = (ax_2_p0, ax_2_p1, ax_2_p2, ax_2_p3, ax_2_p4, ax_2_p5)

            # Waveforms.
            ax_2_w0.plot(
                wave_times, wave_pos, color=wave_colors[-1], linewidth=linew[0]
            )
            ax_2_w1.plot(wave_times, mean_wave_shape, color="k", linewidth=linew[0])
            ax_2_w2.plot(wave_times, wave_neg, color=wave_colors[0], linewidth=linew[0])

            # Ticks, labels.
            for ax_2_p0 in (ax_2_w0, ax_2_w1, ax_2_w2):
                ax_2_p0.spines[["top", "bottom", "left", "right"]].set_visible(False)
                ax_2_p0.set_xticks([])
                ax_2_p0.set_yticks([])

            ax_2_w1.set_ylabel(
                "{} {}".format(comp_title, comp_name),
                fontsize=label_size,
                fontweight="bold",
                rotation=0,
            )

            # Average burst rate and power modulations.
            # (should be drawn once)
            if comp_name == comps_to_vis[0]:
                # Axes initialization.
                ax_2_20 = fig_2.add_subplot(gs_2_2[0])
                ax_2_21 = fig_2.add_subplot(gs_2_2[1])
                ax_2_22 = fig_2.add_subplot(gs_2_2[2])
                ax_2_23 = fig_2.add_subplot(gs_2_2[3])
                ax_2_24 = fig_2.add_subplot(gs_2_2[4])
                ax_2_25 = fig_2.add_subplot(gs_2_2[5])

                ax_2_2 = (ax_2_20, ax_2_21, ax_2_22, ax_2_23, ax_2_24, ax_2_25)

                # Burst rate and power.
                (
                    overall_br_11,
                    overall_br_11_sem,
                    overall_br_12,
                    overall_br_12_sem,
                ) = compute_burst_rate(
                    sub_dict_cond1,
                    trials_cond1,
                    channel_ids,
                    binned_plot_time,
                    baseline_bins,
                    bin_dt,
                )
                (
                    overall_br_21,
                    overall_br_21_sem,
                    overall_br_22,
                    overall_br_22_sem,
                ) = compute_burst_rate(
                    sub_dict_cond2,
                    trials_cond2,
                    channel_ids,
                    binned_plot_time,
                    baseline_bins,
                    bin_dt,
                )

                overall_rates = [
                    overall_br_11,
                    overall_br_12,
                    overall_br_11 - overall_br_12,
                    overall_br_21,
                    overall_br_22,
                    overall_br_21 - overall_br_22,
                ]

                overall_rates_sems = [
                    overall_br_11_sem,
                    overall_br_12_sem,
                    np.std(overall_br_11 - overall_br_12)
                    / np.sqrt(overall_br_11.shape[0]),
                    overall_br_21_sem,
                    overall_br_22_sem,
                    np.std(overall_br_21 - overall_br_22)
                    / np.sqrt(overall_br_21.shape[0]),
                ]

                powers = [
                    epochs_power_dict["epochs_11"],
                    epochs_power_dict["epochs_12"],
                    epochs_power_dict["epochs_11"] - epochs_power_dict["epochs_12"],
                    epochs_power_dict["epochs_21"],
                    epochs_power_dict["epochs_22"],
                    epochs_power_dict["epochs_21"] - epochs_power_dict["epochs_22"],
                ]

                powers_sem = [
                    epochs_power_dict["epochs_11_sem"],
                    epochs_power_dict["epochs_12_sem"],
                    np.std(
                        epochs_power_dict["epochs_11"] - epochs_power_dict["epochs_12"]
                    )
                    / np.sqrt(epochs_power_dict["epochs_11"].shape[0]),
                    epochs_power_dict["epochs_21_sem"],
                    epochs_power_dict["epochs_22_sem"],
                    np.std(
                        epochs_power_dict["epochs_21"] - epochs_power_dict["epochs_22"]
                    )
                    / np.sqrt(epochs_power_dict["epochs_21"].shape[0]),
                ]

                # Plot.
                for br, br_sem, power, power_sem, ax_2_2x in zip(
                    overall_rates, overall_rates_sems, powers, powers_sem, ax_2_2
                ):
                    bag_of_axes.append(ax_2_2x)

                    ax_2_2x.plot(
                        binned_plot_time[:-1],
                        br,
                        color="indianred",
                        linestyle="-",
                        linewidth=linew[3],
                        zorder=1,
                    )
                    ax_2_2x.fill_between(
                        binned_plot_time[:-1],
                        br + br_sem,
                        br - br_sem,
                        color="indianred",
                        alpha=0.2,
                        zorder=1,
                    )

                    ax_2_2x.plot(
                        binned_plot_time[:-1],
                        power,
                        color="slateblue",
                        linestyle="-",
                        linewidth=linew[3],
                        zorder=2,
                    )
                    ax_2_2x.fill_between(
                        binned_plot_time[:-1],
                        power + power_sem,
                        power - power_sem,
                        color="slateblue",
                        alpha=0.2,
                        zorder=2,
                    )

                    ax_2_2x.axvline(
                        task_time_lims[0],
                        linestyle=":",
                        color="tab:gray",
                        linewidth=linew[2],
                        zorder=0,
                    )
                    ax_2_2x.axvline(
                        task_time_lims[1],
                        linestyle=":",
                        color="tab:gray",
                        linewidth=linew[2],
                        zorder=0,
                    )

                    modulation_scores_min.append(np.min(ax_2_2x.get_ylim()))
                    modulation_scores_max.append(np.max(ax_2_2x.get_ylim()))

                # Limits, ticks, labels.
                for k, ax_2_2x in enumerate(ax_2_2):
                    if k == 0:
                        ax_2_2x.set_ylabel(
                            "Δ burst rate & power (%)", fontsize=label_size
                        )

                    ax_2_2x.set_xticks(task_time_lims)
                    ax_2_2x.set_xticklabels(task_time_lims, fontsize=label_size)

                    ax_2_2x.margins(x=0)
                    ax_2_2x.patch.set_alpha(0.0)
                    ax_2_2x.spines[["top", "right"]].set_visible(False)

            # Computation of waveform-group average modulation.
            sb = mh1_1.T.shape[0] // n_collapsed_groups

            for cg, this_group in enumerate(np.arange(0, n_collapsed_groups, 2)):
                start = this_group * sb
                end = start + sb
                if this_group != n_collapsed_groups - 1:
                    split_0 = mh1_1.T[start:end, :]
                    split_1 = mh1_2.T[start:end, :]
                    split_2 = np.mean(
                        mv_burst_rate_cond1_1 - mv_burst_rate_cond1_2, axis=0
                    ).T[start:end, :]
                    split_3 = mh2_1.T[start:end, :]
                    split_4 = mh2_2.T[start:end, :]
                    split_5 = np.mean(
                        mv_burst_rate_cond2_1 - mv_burst_rate_cond2_2, axis=0
                    ).T[start:end, :]
                else:
                    split_0 = mh1_1.T[start:, :]
                    split_1 = mh1_2.T[start:, :]
                    split_2 = np.mean(
                        mv_burst_rate_cond1_1 - mv_burst_rate_cond1_2, axis=0
                    ).T[start:, :]
                    split_3 = mh2_1.T[start:, :]
                    split_4 = mh2_2.T[start:, :]
                    split_5 = np.mean(
                        mv_burst_rate_cond2_1 - mv_burst_rate_cond2_2, axis=0
                    ).T[start:, :]

                splits = [split_0, split_1, split_2, split_3, split_4, split_5]

                for split, ax in zip(splits, ax_2_p):
                    bag_of_axes.append(ax)

                    mod_sem = np.std(split, axis=0) / np.sqrt(split.shape[0])
                    modulation = np.mean(split, axis=0)

                    modulation_scores_min.append(np.min(modulation))
                    modulation_scores_max.append(np.max(modulation))

                    ax.plot(
                        binned_plot_time[:-1],
                        modulation,
                        color=ax_2_colors[cg],
                        linewidth=linew[3],
                        zorder=zorders[cg],
                    )
                    ax.fill_between(
                        binned_plot_time[:-1],
                        modulation + mod_sem,
                        modulation - mod_sem,
                        color=ax_2_colors[cg],
                        zorder=zorders[cg],
                        alpha=0.2,
                    )

                    ax.axvline(
                        task_time_lims[0],
                        linestyle=":",
                        color="tab:gray",
                        linewidth=linew[2],
                        zorder=0,
                    )
                    ax.axvline(
                        task_time_lims[1],
                        linestyle=":",
                        color="tab:gray",
                        linewidth=linew[2],
                        zorder=0,
                    )

            # Axes, ticks, labels.
            for k, ax in enumerate(ax_2_p):
                if k == 0:
                    ax.set_ylabel("Δ burst rate (%)", fontsize=label_size)

                ax.set_xticks(task_time_lims)
                ax.set_xticklabels(task_time_lims, fontsize=label_size)
                if comp_name == comps_to_vis[1]:
                    ax.set_xlabel("Time (s)", fontsize=label_size)

                ax.margins(x=0)
                ax.patch.set_alpha(0.0)
                ax.spines[["top", "right"]].set_visible(False)

        # Y limits.
        if comp_name == comps_to_vis[1]:
            ylabels = [
                int(np.floor(np.min(modulation_scores_min))),
                0,
                int(np.ceil(np.max(modulation_scores_max))),
            ]

            for bid, bax in enumerate(bag_of_axes):
                bax.set_ylim(
                    [
                        np.min(modulation_scores_min) - 5,
                        np.max(modulation_scores_max) + 5,
                    ]
                )
                bax.set_yticks(ylabels, fontsize=label_size)

                if bid % 6 == 0:
                    bax.set_yticklabels(ylabels, fontsize=label_size)
                else:
                    bax.set_yticklabels([])
        # ----- #

        # Axis 4: colorbar.
        min_ticks = [-5, -10, -25, -50, -75, -100, -150, -200, -300, -400]
        min_tick = np.where(minima < min_ticks)[0]
        cb_min = min_ticks[min_tick[-1]]

        max_ticks = [5, 10, 25, 50, 75, 100, 150, 200, 300, 400]
        max_tick = np.where(maxima > max_ticks)[0]
        cb_max = max_ticks[max_tick[-1]]

        cb = fig.colorbar(im11, cax=ax123, ticks=[cb_min, 0.0, cb_max])
        cb.set_label(label="Δ burst rate (%)", fontsize=label_size)
        cb.ax.tick_params(labelsize=label_size)

        # Labels and ticks.
        ax110.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax110.set_xticks([])
        ax110.set_yticks([])

        if comp != n_comps - 1:
            for ax in (ax111, ax112, ax113, ax120, ax121, ax122):
                ax.set_xticks(task_time_lims)
                ax.set_xticklabels([])
        else:
            for ax in (ax111, ax112, ax113, ax120, ax121, ax122):
                ax.set_xticks(task_time_lims)
                ax.set_xticklabels(task_time_lims, fontsize=label_size)
        if comp == n_comps - 1:
            for ax in (ax111, ax112, ax113, ax120, ax121, ax122):
                ax.set_xlabel("Time (s)", fontsize=label_size)

        for ax in (ax112, ax113, ax120, ax121, ax122):
            ax.set_yticklabels([])
        ax111.set_ylabel("Score (a.u.)", fontsize=label_size)

    # ----- #
    # 4. Optional saving.
    if savefigs == True:
        savepath = join(savepath, "sub_{}/".format(subject))
        fig_name = join(
            savepath,
            "{}_{}_band_features_modulation_{}_{}_comps_{}_to_{}.{}".format(
                solver,
                band,
                tf_method,
                ch_name,
                comps_to_analyze[0],
                comps_to_analyze[-1],
                plot_format,
            ),
        )
        fig.savefig(fig_name, dpi=dpi, facecolor="w", edgecolor="w")

        fig_2_name = join(
            savepath,
            "{}_{}_band_{}_collapsed_features_{}.{}".format(
                solver, band, tf_method, ch_name, plot_format
            ),
        )
        fig_2.savefig(fig_2_name, dpi=dpi, facecolor="w", edgecolor="w")

        if show_sample == True:
            sample_name = join(
                savepath,
                "sample_{}_band_features_modulation_pcs_{}_{}.{}".format(
                    band, comps_to_analyze[0], comps_to_analyze[-1], plot_format
                ),
            )
            fig_sample.savefig(sample_name, dpi=dpi, facecolor="w", edgecolor="w")
        plt.close("all")

    elif savefigs == False:
        plt.show()


def features_modulation_nd(
    subject,
    exp_variables,
    sub_dict_cond1,
    sub_dict_cond2,
    sub_dict_trials_cond1,
    sub_dict_trials_cond2,
    channels,
    sub_scores_dists,
    comps_to_vis,
    comps_groups,
    binned_plot_time,
    bin_dt,
    band,
    task_time_lims,
    baseline_time_lims,
    savepath,
    rate_computation="joint",
    apply_baseline="True",
    plot_difference=False,
    screen_res=[1920, 972],
    dpi=300,
    savefigs=True,
    plot_format="pdf",
):
    """
    Figures of the trial-averaged burst modulation across the time and scores axes for two
    channels (most interestingly C3 and C4) per component and experimental condition.

    Parameters
    ----------
    subject : int
              Integer indicating the subjects' data used for creating
              the burst dictionary.
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
    sub_scores_dists: list
                      List containing all burst scores along each principal
                      component axis (based on all trials) for 'subject'.
    comps_to_vis: list or numpy array
                  List of the indices of components used for the visualization.
    comps_groups: int
                  Number of groups for the scores of each component
                  axis should be split into.
    binned_plot_time: numpy array
                      Array representing the trimmed experimental time
                      with wide time steps, needed for an estimation
                      of burst rate in non-overlapping windows.
    bin_dt: float
            Time step for advancing to a new time window.
    band: str {"mu", "beta"}
          Select band for burst detection.
    task_time_lims: two-element list or numpy array
                    Start and end time of the task period (in seconds,
                    relative to the time-locked event).
    baseline_time_lims: two-element list or 1D array
                        Start and end time of the baseline period (in seconds,
                        relative to the time-locked event).
    savepath: str
              Parent directory that contains all results. Ignored if
              'savefigs' is set to "False".
    rate_computation: str {"independent", "joint"}, optional
                      String that controls whether the burst rate is
                      computed indepently along each of the provided
                      'comps_to_vis', or jointly.
                      Defaults to "joint".
    apply_baseline: bool, optional
                    If set to "True" apply baseline correction using
                    the average across trials as baseline.
                    Defaults to "True".
    plot_difference: bool or "condition", optional
                     If "True" plot the difference between C3 and C4 per
                     condition, instead of each channel. If set to
                     "condition" plot the difference of the same channel
                     between conditions.
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
    # Number of available components.
    n_comps = len(comps_to_vis)
    if n_comps != 2:
        raise ValueError(
            "Currently only supporting 2 dimensions! Change the 'comps_to_vis' variable."
        )

    savepath_gif = join(savepath, "sub_{}".format(subject), "comodulation_gif")
    if not exists(savepath_gif):
        makedirs(savepath_gif)

    # Nunber of waveforms groups for subplots.
    n_collapsed_groups = 3

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
    task_period_bins = np.where(
        (binned_plot_time > baseline_time_lims[1])
        & (binned_plot_time <= task_time_lims[1])
    )[0]

    # Binning of burst scores axis and smoothing kernel stds variables.
    score_bins = 41

    # ----- #
    # 2. Figure.
    # Figures initialization.
    if rate_computation == "joint":
        l = 0.10
        r = 0.90
        b = 0.10
        t = 0.90
        ws = 0.15
    elif rate_computation == "independent":
        l = 0.02
        r = 0.98
        b = 0.20
        t = 0.90
        ws = 0.05

        l2d = 0.10
        r2d = 0.90
        b2d = 0.10
        t2d = 0.90
        ws2d = 0.15

    fig = plt.figure(constrained_layout=False, figsize=(7, 4.5), dpi=dpi)

    gs = fig.add_gridspec(nrows=1, ncols=1, left=l, right=r, bottom=b, top=t)
    gs0 = gs[0].subgridspec(
        nrows=n_collapsed_groups, ncols=n_collapsed_groups, wspace=ws
    )

    if rate_computation == "independent":
        fig1 = plt.figure(constrained_layout=False, figsize=(7, 4.5), dpi=dpi)
        fig2 = plt.figure(constrained_layout=False, figsize=(7, 4.5), dpi=dpi)

        gs1 = fig1.add_gridspec(
            nrows=2,
            ncols=2,
            left=l2d,
            right=r2d,
            bottom=b2d,
            top=t2d,
            width_ratios=[4, 1],
            height_ratios=[1, 9],
            wspace=0.05,
            hspace=0.1,
        )
        gs10 = gs1[0, 0].subgridspec(
            nrows=1, ncols=n_collapsed_groups**2, wspace=ws2d
        )
        gs11 = gs1[1, 0].subgridspec(
            nrows=n_collapsed_groups, ncols=n_collapsed_groups, wspace=ws2d
        )
        gs12 = gs1[1, 1].subgridspec(
            nrows=n_collapsed_groups**2, ncols=2, wspace=ws2d
        )

        gs2 = fig2.add_gridspec(
            nrows=2,
            ncols=2,
            left=l2d,
            right=r2d,
            bottom=b2d,
            top=t2d,
            width_ratios=[4, 1],
            height_ratios=[1, 9],
            wspace=0.05,
            hspace=0.1,
        )
        gs20 = gs2[0, 0].subgridspec(nrows=1, ncols=n_collapsed_groups**2, wspace=ws)
        gs21 = gs2[1, 0].subgridspec(
            nrows=n_collapsed_groups, ncols=n_collapsed_groups, wspace=ws
        )
        gs22 = gs2[1, 1].subgridspec(nrows=n_collapsed_groups**2, ncols=2, wspace=ws)

    title_size = 8
    label_size = 5
    tick_size = 5
    linew = [1.0, 0.5, 1.5]
    m_size = 2
    legend_size = 6
    legend_label_size = 5
    wave_colors = plt.cm.cool(np.linspace(0, 1, 5))

    # Control the labels and colors based on whether the difference across
    # channels is preferred or not.
    if plot_difference == False:
        fig_colors = ["firebrick", "royalblue", "darkorange", "darkorchid"]
        fig_labels = [
            "C3, left hand",
            "C4, left hand",
            "C3, right hand",
            "C4, right hand",
        ]
        fig_name_ind = ""

    elif plot_difference == True:
        fig_colors = ["firebrick", "royalblue"]
        fig_labels = ["C3 - C4, left hand", "C3 - C4, right hand"]
        fig_name_ind = "_chan"

    elif plot_difference == "condition":
        fig_colors = ["firebrick", "royalblue"]
        fig_labels = ["C3, left - right hand", "C4, left - right hand"]
        fig_name_ind = "_cond"

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

    # Burst rate computation.
    # Iteratation over components.
    measures_bins = []
    scores_conds1 = []
    scores_conds2 = []
    waves = []
    for comp, comp_name in enumerate(comps_to_vis):
        # ASSUME THAT THE NUMBER OF AVAILABLE COMPONENTS
        # IS AT LEAST EQUAL TO MAX(comps_to_vis)
        scd = sub_scores_dists[comp_name - 1]

        # Common scores limits for all subjects.
        measures_bins.append(np.linspace(np.min(scd), np.max(scd), score_bins))
        scores_conds1.append(scd[np.hstack(sub_dict_trials_cond1)])
        scores_conds2.append(scd[np.hstack(sub_dict_trials_cond2)])

        # "Extreme" waveforms.
        iqrs = np.linspace(np.min(scd), np.max(scd), comps_groups + 1)
        scores_lims = []
        for i in range(comps_groups):
            scores_lims.append([iqrs[i], iqrs[i + 1]])
        scores_lims = np.array(scores_lims)

        ext_neg_1 = np.where(scores_conds1[comp] <= scores_lims[0][1])[0]
        ext_pos_1 = np.where(scores_conds1[comp] >= scores_lims[-1][0])[0]
        ext_neg_2 = np.where(scores_conds2[comp] <= scores_lims[0][1])[0]
        ext_pos_2 = np.where(scores_conds2[comp] >= scores_lims[-1][0])[0]

        wave_neg = np.mean(
            np.vstack(
                (
                    sub_dict_cond1["waveform"][ext_neg_1],
                    sub_dict_cond2["waveform"][ext_neg_2],
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
        waves.append([wave_neg, wave_pos])

    measures_bins.insert(0, binned_plot_time)
    scores_conds1 = np.array(scores_conds1)
    scores_conds2 = np.array(scores_conds2)

    # Burst rate variables initialization.
    if rate_computation == "joint":
        mv_burst_rate_cond_11, mv_burst_rate_cond_12 = compute_burst_rate_nd(
            sub_dict_cond1,
            trials_cond1,
            channel_ids,
            scores_conds1,
            measures_bins,
            binned_plot_time,
            bin_dt,
            data_type=rate_computation,
        )

        mv_burst_rate_cond_21, mv_burst_rate_cond_22 = compute_burst_rate_nd(
            sub_dict_cond2,
            trials_cond2,
            channel_ids,
            scores_conds2,
            measures_bins,
            binned_plot_time,
            bin_dt,
            data_type=rate_computation,
        )

    elif rate_computation == "independent":
        mv_burst_rate_cond_11 = np.zeros(
            (
                len(trials_cond1),
                len(binned_plot_time) - 1,
                len(measures_bins[1]) - 1,
                n_comps,
            )
        )
        mv_burst_rate_cond_12 = np.zeros(
            (
                len(trials_cond1),
                len(binned_plot_time) - 1,
                len(measures_bins[1]) - 1,
                n_comps,
            )
        )
        mv_burst_rate_cond_21 = np.zeros(
            (
                len(trials_cond2),
                len(binned_plot_time) - 1,
                len(measures_bins[1]) - 1,
                n_comps,
            )
        )
        mv_burst_rate_cond_22 = np.zeros(
            (
                len(trials_cond2),
                len(binned_plot_time) - 1,
                len(measures_bins[1]) - 1,
                n_comps,
            )
        )

        for c, _ in enumerate(comps_to_vis):
            mv_burst_rate_cond1_1, mv_burst_rate_cond1_2 = compute_burst_rate_nd(
                sub_dict_cond1,
                trials_cond1,
                channel_ids,
                scores_conds1[c],
                measures_bins[1 + c],
                binned_plot_time,
                bin_dt,
                data_type=rate_computation,
            )
            mv_burst_rate_cond2_1, mv_burst_rate_cond2_2 = compute_burst_rate_nd(
                sub_dict_cond2,
                trials_cond2,
                channel_ids,
                scores_conds2[c],
                measures_bins[1 + c],
                binned_plot_time,
                bin_dt,
                data_type=rate_computation,
            )

            mv_burst_rate_cond_11[:, :, :, c] = mv_burst_rate_cond1_1
            mv_burst_rate_cond_12[:, :, :, c] = mv_burst_rate_cond1_2
            mv_burst_rate_cond_21[:, :, :, c] = mv_burst_rate_cond2_1
            mv_burst_rate_cond_22[:, :, :, c] = mv_burst_rate_cond2_2

    # Baseline correction.
    if rate_computation == "joint":
        mhs_11 = np.zeros(
            [len(binned_plot_time) - 1, n_collapsed_groups, n_collapsed_groups]
        )
        mhs_12 = np.zeros(
            [len(binned_plot_time) - 1, n_collapsed_groups, n_collapsed_groups]
        )
        mhs_21 = np.zeros(
            [len(binned_plot_time) - 1, n_collapsed_groups, n_collapsed_groups]
        )
        mhs_22 = np.zeros(
            [len(binned_plot_time) - 1, n_collapsed_groups, n_collapsed_groups]
        )

        # Optional baseline correction.
        if apply_baseline == True:
            mhc_11 = np.mean(mv_burst_rate_cond_11[:, baseline_bins, :], axis=(0, 1))
            mhc_12 = np.mean(mv_burst_rate_cond_12[:, baseline_bins, :], axis=(0, 1))
            mhc_21 = np.mean(mv_burst_rate_cond_21[:, baseline_bins, :], axis=(0, 1))
            mhc_22 = np.mean(mv_burst_rate_cond_22[:, baseline_bins, :], axis=(0, 1))

            mv_burst_rate_cond_11_temp = (mv_burst_rate_cond_11 - mhc_11) / mhc_11 * 100
            mv_burst_rate_cond_12_temp = (mv_burst_rate_cond_12 - mhc_12) / mhc_12 * 100
            mv_burst_rate_cond_21_temp = (mv_burst_rate_cond_21 - mhc_21) / mhc_21 * 100
            mv_burst_rate_cond_22_temp = (mv_burst_rate_cond_22 - mhc_22) / mhc_22 * 100

            # Average over trials.
            mh_11 = np.mean(mv_burst_rate_cond_11_temp, axis=0)
            mh_12 = np.mean(mv_burst_rate_cond_12_temp, axis=0)
            mh_21 = np.mean(mv_burst_rate_cond_21_temp, axis=0)
            mh_22 = np.mean(mv_burst_rate_cond_22_temp, axis=0)

        else:
            # Average over trials.
            mh_11 = np.mean(mv_burst_rate_cond_11, axis=0)
            mh_12 = np.mean(mv_burst_rate_cond_12, axis=0)
            mh_21 = np.mean(mv_burst_rate_cond_21, axis=0)
            mh_22 = np.mean(mv_burst_rate_cond_22, axis=0)

        # Computation of waveform-group average modulation.
        sb = mh_11.shape[-1] // comps_groups

        for cg, this_group in enumerate(np.arange(0, comps_groups, 2)):
            start = this_group * sb
            end = start + sb

            for cg_in, this_group_in in enumerate(np.arange(0, comps_groups, 2)):
                start_in = this_group_in * sb
                end_in = start_in + sb

                if this_group != comps_groups - 1:
                    if this_group_in != comps_groups - 1:
                        mhs_11[:, cg, cg_in] = np.mean(
                            mh_11[:, start:end, start_in:end_in], axis=(1, 2)
                        )
                        mhs_12[:, cg, cg_in] = np.mean(
                            mh_12[:, start:end, start_in:end_in], axis=(1, 2)
                        )
                        mhs_21[:, cg, cg_in] = np.mean(
                            mh_21[:, start:end, start_in:end_in], axis=(1, 2)
                        )
                        mhs_22[:, cg, cg_in] = np.mean(
                            mh_22[:, start:end, start_in:end_in], axis=(1, 2)
                        )
                    else:
                        mhs_11[:, cg, cg_in] = np.mean(
                            mh_11[:, start:end, start_in:], axis=(1, 2)
                        )
                        mhs_12[:, cg, cg_in] = np.mean(
                            mh_12[:, start:end, start_in:], axis=(1, 2)
                        )
                        mhs_21[:, cg, cg_in] = np.mean(
                            mh_21[:, start:end, start_in:], axis=(1, 2)
                        )
                        mhs_22[:, cg, cg_in] = np.mean(
                            mh_22[:, start:end, start_in:], axis=(1, 2)
                        )
                else:
                    if this_group_in != comps_groups - 1:
                        mhs_11[:, cg, cg_in] = np.mean(
                            mh_11[:, start:, start_in:end_in], axis=(1, 2)
                        )
                        mhs_12[:, cg, cg_in] = np.mean(
                            mh_12[:, start:, start_in:end_in], axis=(1, 2)
                        )
                        mhs_21[:, cg, cg_in] = np.mean(
                            mh_21[:, start:, start_in:end_in], axis=(1, 2)
                        )
                        mhs_22[:, cg, cg_in] = np.mean(
                            mh_22[:, start:, start_in:end_in], axis=(1, 2)
                        )
                    else:
                        mhs_11[:, cg, cg_in] = np.mean(
                            mh_11[:, start:, start_in:], axis=(1, 2)
                        )
                        mhs_12[:, cg, cg_in] = np.mean(
                            mh_12[:, start:, start_in:], axis=(1, 2)
                        )
                        mhs_21[:, cg, cg_in] = np.mean(
                            mh_21[:, start:, start_in:], axis=(1, 2)
                        )
                        mhs_22[:, cg, cg_in] = np.mean(
                            mh_22[:, start:, start_in:], axis=(1, 2)
                        )

    elif rate_computation == "independent":
        mhs_11 = np.zeros([n_comps, len(binned_plot_time) - 1, n_collapsed_groups])
        mhs_12 = np.zeros([n_comps, len(binned_plot_time) - 1, n_collapsed_groups])
        mhs_21 = np.zeros([n_comps, len(binned_plot_time) - 1, n_collapsed_groups])
        mhs_22 = np.zeros([n_comps, len(binned_plot_time) - 1, n_collapsed_groups])

        mhs_11_t = np.zeros([n_comps, len(trials_cond1), n_collapsed_groups])
        mhs_12_t = np.zeros([n_comps, len(trials_cond1), n_collapsed_groups])
        mhs_21_t = np.zeros([n_comps, len(trials_cond2), n_collapsed_groups])
        mhs_22_t = np.zeros([n_comps, len(trials_cond2), n_collapsed_groups])

        mhs_11_tt = np.zeros(
            [n_comps, len(trials_cond1), len(binned_plot_time) - 1, n_collapsed_groups]
        )
        mhs_12_tt = np.zeros(
            [n_comps, len(trials_cond1), len(binned_plot_time) - 1, n_collapsed_groups]
        )
        mhs_21_tt = np.zeros(
            [n_comps, len(trials_cond2), len(binned_plot_time) - 1, n_collapsed_groups]
        )
        mhs_22_tt = np.zeros(
            [n_comps, len(trials_cond2), len(binned_plot_time) - 1, n_collapsed_groups]
        )

        for j in range(n_comps):
            # Optional baseline correction.
            if apply_baseline == True:
                mhc_11 = np.mean(
                    mv_burst_rate_cond_11[:, baseline_bins, :, j], axis=(0, 1)
                )
                mhc_12 = np.mean(
                    mv_burst_rate_cond_12[:, baseline_bins, :, j], axis=(0, 1)
                )
                mhc_21 = np.mean(
                    mv_burst_rate_cond_21[:, baseline_bins, :, j], axis=(0, 1)
                )
                mhc_22 = np.mean(
                    mv_burst_rate_cond_22[:, baseline_bins, :, j], axis=(0, 1)
                )

                mv_burst_rate_cond_11_temp = (
                    (mv_burst_rate_cond_11[:, :, :, j] - mhc_11) / mhc_11 * 100
                )
                mv_burst_rate_cond_12_temp = (
                    (mv_burst_rate_cond_12[:, :, :, j] - mhc_12) / mhc_12 * 100
                )
                mv_burst_rate_cond_21_temp = (
                    (mv_burst_rate_cond_21[:, :, :, j] - mhc_21) / mhc_21 * 100
                )
                mv_burst_rate_cond_22_temp = (
                    (mv_burst_rate_cond_22[:, :, :, j] - mhc_22) / mhc_22 * 100
                )

                # Average over trials.
                mh_11 = np.mean(mv_burst_rate_cond_11_temp, axis=0)
                mh_12 = np.mean(mv_burst_rate_cond_12_temp, axis=0)
                mh_21 = np.mean(mv_burst_rate_cond_21_temp, axis=0)
                mh_22 = np.mean(mv_burst_rate_cond_22_temp, axis=0)

                # Average over time.
                mh_11_t = np.mean(
                    mv_burst_rate_cond_11_temp[:, task_period_bins, :], axis=1
                )
                mh_12_t = np.mean(
                    mv_burst_rate_cond_12_temp[:, task_period_bins, :], axis=1
                )
                mh_21_t = np.mean(
                    mv_burst_rate_cond_21_temp[:, task_period_bins, :], axis=1
                )
                mh_22_t = np.mean(
                    mv_burst_rate_cond_22_temp[:, task_period_bins, :], axis=1
                )

            else:
                # Average over trials.
                mh_11 = np.mean(mv_burst_rate_cond_11[:, :, :, j], axis=0)
                mh_12 = np.mean(mv_burst_rate_cond_12[:, :, :, j], axis=0)
                mh_21 = np.mean(mv_burst_rate_cond_21[:, :, :, j], axis=0)
                mh_22 = np.mean(mv_burst_rate_cond_22[:, :, :, j], axis=0)

                # Average over time.
                mh_11_t = np.mean(
                    mv_burst_rate_cond_11[:, task_period_bins, :][:, :, :, j], axis=1
                )
                mh_12_t = np.mean(
                    mv_burst_rate_cond_12[:, task_period_bins, :][:, :, :, j], axis=1
                )
                mh_21_t = np.mean(
                    mv_burst_rate_cond_21[:, task_period_bins, :][:, :, :, j], axis=1
                )
                mh_22_t = np.mean(
                    mv_burst_rate_cond_22[:, task_period_bins, :][:, :, :, j], axis=1
                )

            # Computation of waveform-group average modulation.
            sb = mh_11.shape[-1] // comps_groups

            for cg, this_group in enumerate(np.arange(0, comps_groups, 2)):
                start = this_group * sb
                end = start + sb

                if this_group != comps_groups - 1:
                    mhs_11[j, :, cg] = np.mean(mh_11[:, start:end], axis=-1)
                    mhs_12[j, :, cg] = np.mean(mh_12[:, start:end], axis=-1)
                    mhs_21[j, :, cg] = np.mean(mh_21[:, start:end], axis=-1)
                    mhs_22[j, :, cg] = np.mean(mh_22[:, start:end], axis=-1)

                    mhs_11_t[j, :, cg] = np.mean(mh_11_t[:, start:end], axis=-1)
                    mhs_12_t[j, :, cg] = np.mean(mh_12_t[:, start:end], axis=-1)
                    mhs_21_t[j, :, cg] = np.mean(mh_21_t[:, start:end], axis=-1)
                    mhs_22_t[j, :, cg] = np.mean(mh_22_t[:, start:end], axis=-1)

                    mhs_11_tt[j, :, :, cg] = np.mean(
                        mv_burst_rate_cond_11_temp[:, :, start:end], axis=-1
                    )
                    mhs_12_tt[j, :, :, cg] = np.mean(
                        mv_burst_rate_cond_12_temp[:, :, start:end], axis=-1
                    )
                    mhs_21_tt[j, :, :, cg] = np.mean(
                        mv_burst_rate_cond_21_temp[:, :, start:end], axis=-1
                    )
                    mhs_22_tt[j, :, :, cg] = np.mean(
                        mv_burst_rate_cond_22_temp[:, :, start:end], axis=-1
                    )

                else:
                    mhs_11[j, :, cg] = np.mean(mh_11[:, start:], axis=-1)
                    mhs_12[j, :, cg] = np.mean(mh_12[:, start:], axis=-1)
                    mhs_21[j, :, cg] = np.mean(mh_21[:, start:], axis=-1)
                    mhs_22[j, :, cg] = np.mean(mh_22[:, start:], axis=-1)

                    mhs_11_t[j, :, cg] = np.mean(mh_11_t[:, start:], axis=-1)
                    mhs_12_t[j, :, cg] = np.mean(mh_12_t[:, start:], axis=-1)
                    mhs_21_t[j, :, cg] = np.mean(mh_21_t[:, start:], axis=-1)
                    mhs_22_t[j, :, cg] = np.mean(mh_22_t[:, start:], axis=-1)

                    mhs_11_tt[j, :, :, cg] = np.mean(
                        mv_burst_rate_cond_11_temp[:, :, start:], axis=-1
                    )
                    mhs_12_tt[j, :, :, cg] = np.mean(
                        mv_burst_rate_cond_12_temp[:, :, start:], axis=-1
                    )
                    mhs_21_tt[j, :, :, cg] = np.mean(
                        mv_burst_rate_cond_21_temp[:, :, start:], axis=-1
                    )
                    mhs_22_tt[j, :, :, cg] = np.mean(
                        mv_burst_rate_cond_22_temp[:, :, start:], axis=-1
                    )

    # 3. Plotting.

    # Waveform shapes.
    if rate_computation == "independent":
        for fig12, gsx0, gsx2 in zip([fig1, fig2], [gs10, gs20], [gs12, gs22]):
            ax_n1 = fig12.add_subplot(gsx0[1])
            ax_m1 = fig12.add_subplot(gsx0[int(((n_collapsed_groups * 3) - 1) / 2)])
            ax_p1 = fig12.add_subplot(gsx0[-2])

            ax_n1.plot(
                wave_times, waves[0][0], color=wave_colors[0], linewidth=linew[0]
            )
            ax_m1.plot(wave_times, mean_wave_shape, color="k", linewidth=linew[0])
            ax_p1.plot(
                wave_times, waves[0][1], color=wave_colors[-1], linewidth=linew[0]
            )

            ax_n2 = fig12.add_subplot(gsx2[-2, 0])
            ax_m2 = fig12.add_subplot(gsx2[int(((n_collapsed_groups * 3) - 1) / 2), 0])
            ax_p2 = fig12.add_subplot(gsx2[1, 0])

            ax_n2.plot(
                wave_times, waves[1][0], color=wave_colors[0], linewidth=linew[0]
            )
            ax_m2.plot(wave_times, mean_wave_shape, color="k", linewidth=linew[0])
            ax_p2.plot(
                wave_times, waves[1][1], color=wave_colors[-1], linewidth=linew[0]
            )

            for ax_nmp in [ax_n1, ax_m1, ax_p1, ax_n2, ax_m2, ax_p2]:
                ax_nmp.spines[["top", "bottom", "left", "right"]].set_visible(False)
                ax_nmp.set_xticks([])
                ax_nmp.set_yticks([])

    # Burst rates.
    for aa in range(n_collapsed_groups**2):
        # Data selection.
        id_x = aa % n_collapsed_groups
        id_y = aa // n_collapsed_groups

        # Axes selection.
        id_x_plot = np.abs(id_y - (n_collapsed_groups - 1))
        id_y_plot = id_x

        if rate_computation == "joint":
            ax = fig.add_subplot(gs0[id_x_plot, id_y_plot])

            if plot_difference == False:
                data_to_plot = [
                    mhs_11[:, id_x, id_y],
                    mhs_12[:, id_x, id_y],
                    mhs_21[:, id_x, id_y],
                    mhs_22[:, id_x, id_y],
                ]
            elif plot_difference == True:
                data_to_plot = [
                    mhs_11[:, id_x, id_y] - mhs_12[:, id_x, id_y],
                    mhs_21[:, id_x, id_y] - mhs_22[:, id_x, id_y],
                ]
            elif plot_difference == "condition":
                data_to_plot = [
                    mhs_11[:, id_x, id_y] - mhs_21[:, id_x, id_y],
                    mhs_12[:, id_x, id_y] - mhs_22[:, id_x, id_y],
                ]

            # Lines connecting time points.
            for d, data in enumerate(data_to_plot):
                if aa == 0:
                    ax.plot(
                        binned_plot_time[:-1],
                        data,
                        lw=linew[0],
                        c=fig_colors[d],
                        label=fig_labels[d],
                    )
                else:
                    ax.plot(binned_plot_time[:-1], data, lw=linew[0], c=fig_colors[d])

            ax.axvline(task_time_lims[0], linestyle=":", color="k", linewidth=linew[1])
            ax.axvline(task_time_lims[1], linestyle=":", color="k", linewidth=linew[1])

            # Axes, labels, titles, legend.
            if id_y_plot == 0:
                ax.set_ylabel("Δ burst rate (%)", fontsize=label_size)
            if id_x_plot == n_collapsed_groups - 1:
                ax.set_xlabel("Time (s)", fontsize=label_size)

            if id_y_plot != 0:
                ax.set_yticklabels([])
            if id_x_plot != n_collapsed_groups - 1:
                ax.set_xticklabels([])

            if plot_difference == False:
                ax.set_ylim(
                    [
                        np.min([mhs_11, mhs_12, mhs_21, mhs_22]) - 5,
                        np.max([mhs_11, mhs_12, mhs_21, mhs_22]) + 5,
                    ]
                )
            elif plot_difference == True:
                ax.set_ylim(
                    [
                        np.min([mhs_11 - mhs_12, mhs_21 - mhs_22]) - 5,
                        np.max([mhs_11 - mhs_12, mhs_21 - mhs_22]) + 5,
                    ]
                )

            ax.tick_params(axis="x", labelsize=tick_size)
            ax.tick_params(axis="y", labelsize=tick_size)

            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["top", "right"]].set_visible(False)

        elif rate_computation == "independent":
            ax = fig.add_subplot(gs0[id_x_plot, id_y_plot], projection="3d")
            ax1 = fig1.add_subplot(gs11[id_x_plot, id_y_plot])
            ax2 = fig2.add_subplot(gs21[id_x_plot, id_y_plot])

            if plot_difference == False:
                data_to_plot_x = [
                    mhs_11[0, :, id_x],
                    mhs_12[0, :, id_x],
                    mhs_21[0, :, id_x],
                    mhs_22[0, :, id_x],
                ]
                data_to_plot_y = [
                    mhs_11[1, :, id_y],
                    mhs_12[1, :, id_y],
                    mhs_21[1, :, id_y],
                    mhs_22[1, :, id_y],
                ]
                trials_to_scat_x = [
                    mhs_11_t[0, :, id_x],
                    mhs_12_t[0, :, id_x],
                    mhs_21_t[0, :, id_x],
                    mhs_22_t[0, :, id_x],
                ]
                trials_to_scat_y = [
                    mhs_11_t[1, :, id_y],
                    mhs_12_t[1, :, id_y],
                    mhs_21_t[1, :, id_y],
                    mhs_22_t[1, :, id_y],
                ]

            elif plot_difference == True:
                data_to_plot_x = [
                    mhs_11[0, :, id_x] - mhs_12[0, :, id_x],
                    mhs_21[0, :, id_x] - mhs_22[0, :, id_x],
                ]
                data_to_plot_y = [
                    mhs_11[1, :, id_y] - mhs_12[1, :, id_y],
                    mhs_21[1, :, id_y] - mhs_22[1, :, id_y],
                ]
                trials_to_scat_x = [
                    mhs_11_t[0, :, id_x] - mhs_12_t[0, :, id_x],
                    mhs_21_t[0, :, id_x] - mhs_22_t[0, :, id_x],
                ]
                trials_to_scat_y = [
                    mhs_11_t[1, :, id_y] - mhs_12_t[1, :, id_y],
                    mhs_21_t[1, :, id_y] - mhs_22_t[1, :, id_y],
                ]

            elif plot_difference == "condition":
                min_av_trials = np.min(
                    [mhs_11.shape[1], mhs_12.shape[1], mhs_21.shape[1], mhs_22.shape[1]]
                )

                data_to_plot_x = [
                    mhs_11[0, :, id_x] - mhs_21[0, :, id_x],
                    mhs_12[0, :, id_x] - mhs_22[0, :, id_x],
                ]
                data_to_plot_y = [
                    mhs_11[1, :, id_y] - mhs_21[1, :, id_y],
                    mhs_12[1, :, id_y] - mhs_22[1, :, id_y],
                ]
                trials_to_scat_x = [
                    mhs_11_t[0, :min_av_trials, id_x]
                    - mhs_21_t[0, :min_av_trials, id_x],
                    mhs_12_t[0, :min_av_trials, id_x]
                    - mhs_22_t[0, :min_av_trials, id_x],
                ]
                trials_to_scat_y = [
                    mhs_11_t[1, :min_av_trials, id_y]
                    - mhs_21_t[1, :min_av_trials, id_y],
                    mhs_12_t[1, :min_av_trials, id_y]
                    - mhs_22_t[1, :min_av_trials, id_y],
                ]

            # Lines connecting time points.
            for d, (data_x, data_y) in enumerate(zip(data_to_plot_x, data_to_plot_y)):
                if aa == 0:
                    ax.plot(
                        binned_plot_time[:-1],
                        data_x,
                        data_y,
                        lw=linew[0],
                        c=fig_colors[d],
                        label=fig_labels[d],
                    )
                    ax1.plot(
                        data_x[task_period_bins],
                        data_y[task_period_bins],
                        lw=linew[0],
                        c=fig_colors[d],
                        label=fig_labels[d],
                    )
                else:
                    ax.plot(
                        binned_plot_time[:-1],
                        data_x,
                        data_y,
                        lw=linew[0],
                        c=fig_colors[d],
                    )
                    ax1.plot(
                        data_x[task_period_bins],
                        data_y[task_period_bins],
                        lw=linew[0],
                        c=fig_colors[d],
                    )

            # Scatter plot of trials with confidence ellipses.
            # if plot_difference != "condition":
            for d, (data_x, data_y) in enumerate(
                zip(trials_to_scat_x, trials_to_scat_y)
            ):
                if aa == 0:
                    ax2.scatter(
                        data_x,
                        data_y,
                        facecolors=fig_colors[d],
                        s=0.2,
                        marker="o",
                        lw=linew[1],
                        label=fig_labels[d],
                    )
                else:
                    ax2.scatter(
                        data_x,
                        data_y,
                        facecolors=fig_colors[d],
                        s=0.2,
                        marker="o",
                        lw=linew[1],
                    )

                confidence_ellipse(
                    data_x, data_y, ax2, n_std=2.0, lw=linew[2], edgecolor=fig_colors[d]
                )

            # Scatter plots in both cases.
            for t, tp in enumerate(binned_plot_time[:-1]):
                if tp < baseline_time_lims[1]:
                    marker_shape = "^"
                elif tp >= baseline_time_lims[1] and tp < task_time_lims[-1]:
                    marker_shape = "o"
                elif tp >= task_time_lims[-1]:
                    marker_shape = "s"

                if plot_difference == False:
                    data_to_scat_x = [
                        mhs_11[0, t, id_x],
                        mhs_12[0, t, id_x],
                        mhs_21[0, t, id_x],
                        mhs_22[0, t, id_x],
                    ]
                    data_to_scat_y = [
                        mhs_11[1, t, id_y],
                        mhs_12[1, t, id_y],
                        mhs_21[1, t, id_y],
                        mhs_22[1, t, id_y],
                    ]
                elif plot_difference == True:
                    data_to_scat_x = [
                        mhs_11[0, t, id_x] - mhs_12[0, t, id_x],
                        mhs_21[0, t, id_x] - mhs_22[0, t, id_x],
                    ]
                    data_to_scat_y = [
                        mhs_11[1, t, id_y] - mhs_12[1, t, id_y],
                        mhs_21[1, t, id_y] - mhs_22[1, t, id_y],
                    ]
                elif plot_difference == "condition":
                    data_to_scat_x = [
                        mhs_11[0, t, id_x] - mhs_21[0, t, id_x],
                        mhs_12[0, t, id_x] - mhs_22[0, t, id_x],
                    ]
                    data_to_scat_y = [
                        mhs_11[1, t, id_y] - mhs_21[1, t, id_y],
                        mhs_12[1, t, id_y] - mhs_22[1, t, id_y],
                    ]

                for d, (data_x, data_y) in enumerate(
                    zip(data_to_scat_x, data_to_scat_y)
                ):
                    ax.scatter(
                        tp,
                        data_x,
                        data_y,
                        facecolors="None",
                        s=m_size,
                        marker=marker_shape,
                        edgecolors=fig_colors[d],
                        lw=linew[1],
                    )

            # Axes, labels, titles, legend.
            if id_y_plot == 0 and id_x_plot == n_collapsed_groups - 1:
                ax.set_ylabel(
                    "Comp. {} Δ burst rate (%)".format(comps_to_vis[0]),
                    fontsize=label_size,
                )
            if (
                id_y_plot == n_collapsed_groups - 1
                and id_x_plot == n_collapsed_groups - 1
            ):
                ax.set_zlabel(
                    "Comp. {} Δ burst rate (%)".format(comps_to_vis[1]),
                    fontsize=label_size,
                )
            if id_x_plot == n_collapsed_groups - 1:
                ax.set_xlabel("Time (s)", fontsize=label_size)

            if id_y_plot != 0:
                ax.set_yticklabels([])
                ax.set_zticklabels([])
            if id_x_plot != n_collapsed_groups - 1:
                ax.set_xticklabels([])

            if plot_difference == False:
                ax_min = np.min([mhs_11, mhs_12, mhs_21, mhs_22]) - 5
                ax_max = np.max([mhs_11, mhs_12, mhs_21, mhs_22]) + 5
                ax1_min = np.min([mhs_11, mhs_12, mhs_21, mhs_22]) - 5
                ax1_max = np.max([mhs_11, mhs_12, mhs_21, mhs_22]) + 5
                ax2_min = (
                    np.min([np.min([mhs_11_t, mhs_12_t]), np.min([mhs_21_t, mhs_22_t])])
                    * 1.5
                )
                ax2_max = (
                    np.max([np.max([mhs_11_t, mhs_12_t]), np.max([mhs_21_t, mhs_22_t])])
                    * 1.1
                )

            elif plot_difference == True:
                ax_min = np.min([mhs_11 - mhs_12, mhs_21 - mhs_22]) - 5
                ax_max = np.max([mhs_11 - mhs_12, mhs_21 - mhs_22]) + 5
                ax1_min = np.min([mhs_11 - mhs_12, mhs_21 - mhs_22]) - 5
                ax1_max = np.max([mhs_11 - mhs_12, mhs_21 - mhs_22]) + 5
                ax2_min = (
                    np.min(
                        [np.min([mhs_11_t - mhs_12_t]), np.min([mhs_21_t - mhs_22_t])]
                    )
                    * 1.1
                )
                ax2_max = (
                    np.max(
                        [np.max([mhs_11_t - mhs_12_t]), np.max([mhs_21_t - mhs_22_t])]
                    )
                    * 1.1
                )

            ax.set_ylim([ax_min, ax_max])
            ax.set_zlim([ax_min, ax_max])
            ax1.set_xlim([ax1_min, ax1_max])
            ax1.set_ylim([ax1_min, ax1_max])
            ax2.set_xlim([ax2_min, ax2_max])
            ax2.set_ylim([ax2_min, ax2_max])

            if id_y_plot != 0:
                ax1.set_yticklabels([])
                ax2.set_yticklabels([])
            if id_x_plot != n_collapsed_groups - 1:
                ax1.set_xticklabels([])
                ax2.set_xticklabels([])

            ax.tick_params(axis="x", labelsize=tick_size)
            ax.tick_params(axis="y", labelsize=tick_size)
            ax.tick_params(axis="z", labelsize=tick_size)
            ax1.tick_params(axis="x", labelsize=tick_size)
            ax1.tick_params(axis="y", labelsize=tick_size)
            ax2.tick_params(axis="x", labelsize=tick_size)
            ax2.tick_params(axis="y", labelsize=tick_size)

            ax1.spines[["top", "right"]].set_visible(False)
            ax2.spines[["top", "right"]].set_visible(False)

            # Axes, labels, titles, legend.
            if id_y_plot == 0 and id_x_plot == int(n_collapsed_groups / 2):
                ax1.set_ylabel(
                    "Comp. {} Δ burst rate (%)".format(comps_to_vis[1]),
                    fontsize=label_size,
                )
                ax2.set_ylabel(
                    "Comp. {} average Δ burst rate (%)".format(comps_to_vis[1]),
                    fontsize=label_size,
                )

            if id_x_plot == n_collapsed_groups - 1:
                ax1.set_xlabel(
                    "Comp. {} Δ burst rate (%)".format(comps_to_vis[0]),
                    fontsize=label_size,
                )
                ax2.set_xlabel(
                    "Comp. {} average Δ burst rate (%)".format(comps_to_vis[0]),
                    fontsize=label_size,
                )

    # Animation.
    if rate_computation == "independent":
        bag_of_figs3 = []

        for t, tp in enumerate(binned_plot_time[:-1]):
            # Figure initialization.
            fig3 = plt.figure(constrained_layout=False, figsize=(7, 4.5), dpi=dpi)
            gs3 = fig3.add_gridspec(
                nrows=2,
                ncols=2,
                left=l2d,
                right=r2d,
                bottom=b2d,
                top=t2d,
                width_ratios=[4, 1],
                height_ratios=[1, 9],
                wspace=0.05,
                hspace=0.1,
            )
            gs30 = gs3[0, 0].subgridspec(
                nrows=1, ncols=n_collapsed_groups**2, wspace=ws
            )
            gs31 = gs3[1, 0].subgridspec(
                nrows=n_collapsed_groups, ncols=n_collapsed_groups, wspace=ws
            )
            gs32 = gs3[1, 1].subgridspec(
                nrows=n_collapsed_groups**2, ncols=2, wspace=ws
            )

            # Waveform shapes.
            ax_n1 = fig3.add_subplot(gs30[1])
            ax_m1 = fig3.add_subplot(gs30[int(((n_collapsed_groups * 3) - 1) / 2)])
            ax_p1 = fig3.add_subplot(gs30[-2])

            ax_n1.plot(
                wave_times, waves[0][0], color=wave_colors[0], linewidth=linew[0]
            )
            ax_m1.plot(wave_times, mean_wave_shape, color="k", linewidth=linew[0])
            ax_p1.plot(
                wave_times, waves[0][1], color=wave_colors[-1], linewidth=linew[0]
            )

            ax_n2 = fig3.add_subplot(gs32[-2, 0])
            ax_m2 = fig3.add_subplot(gs32[int(((n_collapsed_groups * 3) - 1) / 2), 0])
            ax_p2 = fig3.add_subplot(gs32[1, 0])

            ax_n2.plot(
                wave_times, waves[1][0], color=wave_colors[0], linewidth=linew[0]
            )
            ax_m2.plot(wave_times, mean_wave_shape, color="k", linewidth=linew[0])
            ax_p2.plot(
                wave_times, waves[1][1], color=wave_colors[-1], linewidth=linew[0]
            )

            for ax_nmp in [ax_n1, ax_m1, ax_p1, ax_n2, ax_m2, ax_p2]:
                ax_nmp.spines[["top", "bottom", "left", "right"]].set_visible(False)
                ax_nmp.set_xticks([])
                ax_nmp.set_yticks([])

            # Plotting.
            for aa in range(n_collapsed_groups**2):
                # Data selection.
                id_x = aa % n_collapsed_groups
                id_y = aa // n_collapsed_groups

                # Axes selection.
                id_x_plot = np.abs(id_y - (n_collapsed_groups - 1))
                id_y_plot = id_x

                # Subplot selection.
                ax3 = fig3.add_subplot(gs31[id_x_plot, id_y_plot])

                # Distinct markers per trial period.
                if tp < baseline_time_lims[1]:
                    marker_shape = "^"
                elif tp >= baseline_time_lims[1] and tp < task_time_lims[-1]:
                    marker_shape = "o"
                elif tp >= task_time_lims[-1]:
                    marker_shape = "s"

                # Data.
                if plot_difference == False:
                    data_to_scat_x = [
                        mhs_11_tt[0, :, t, id_x],
                        mhs_12_tt[0, :, t, id_x],
                        mhs_21_tt[0, :, t, id_x],
                        mhs_22_tt[0, :, t, id_x],
                    ]
                    data_to_scat_y = [
                        mhs_11_tt[1, :, t, id_y],
                        mhs_12_tt[1, :, t, id_y],
                        mhs_21_tt[1, :, t, id_y],
                        mhs_22_tt[1, :, t, id_y],
                    ]
                elif plot_difference == True:
                    data_to_scat_x = [
                        mhs_11_tt[0, :, t, id_x] - mhs_12_tt[0, :, t, id_x],
                        mhs_21_tt[0, :, t, id_x] - mhs_22_tt[0, :, t, id_x],
                    ]
                    data_to_scat_y = [
                        mhs_11_tt[1, :, t, id_y] - mhs_12_tt[1, :, t, id_y],
                        mhs_21_tt[1, :, t, id_y] - mhs_22_tt[1, :, t, id_y],
                    ]
                elif plot_difference == "condition":
                    data_to_scat_x = [
                        mhs_11_tt[0, :, t, id_x] - mhs_21_tt[0, :, t, id_x],
                        mhs_12_tt[0, :, t, id_x] - mhs_22_tt[0, :, t, id_x],
                    ]
                    data_to_scat_y = [
                        mhs_11_tt[1, :, t, id_y] - mhs_21_tt[1, :, t, id_y],
                        mhs_12_tt[1, :, t, id_y] - mhs_22_tt[1, :, t, id_y],
                    ]

                # Plots.
                for d, (data_x, data_y) in enumerate(
                    zip(data_to_scat_x, data_to_scat_y)
                ):
                    if aa == 0:
                        scat3 = ax3.scatter(
                            data_x,
                            data_y,
                            facecolors=fig_colors[d],
                            s=0.2,
                            marker=marker_shape,
                            lw=linew[1],
                            label=fig_labels[d],
                        )

                    else:
                        scat3 = ax3.scatter(
                            data_x,
                            data_y,
                            facecolors=fig_colors[d],
                            s=0.2,
                            marker=marker_shape,
                            lw=linew[1],
                        )
                    ellipse3 = confidence_ellipse(
                        data_x,
                        data_y,
                        ax3,
                        n_std=2.0,
                        lw=linew[2],
                        edgecolor=fig_colors[d],
                    )

                # Axes, ticks, labels, legend.
                if plot_difference == False:
                    ax3_min = (
                        np.min(
                            [np.min([mhs_11_t, mhs_12_t]), np.min([mhs_21_t, mhs_22_t])]
                        )
                        * 1.5
                    )
                    ax3_max = (
                        np.max(
                            [np.max([mhs_11_t, mhs_12_t]), np.max([mhs_21_t, mhs_22_t])]
                        )
                        * 1.1
                    )

                elif plot_difference == True:
                    ax3_min = (
                        np.min(
                            [
                                np.min([mhs_11_t - mhs_12_t]),
                                np.min([mhs_21_t - mhs_22_t]),
                            ]
                        )
                        * 1.1
                    )
                    ax3_max = (
                        np.max(
                            [
                                np.max([mhs_11_t - mhs_12_t]),
                                np.max([mhs_21_t - mhs_22_t]),
                            ]
                        )
                        * 1.1
                    )

                ax3.set_xlim([ax3_min, ax3_max])
                ax3.set_ylim([ax3_min, ax3_max])

                if id_y_plot != 0:
                    ax3.set_yticklabels([])
                if id_x_plot != n_collapsed_groups - 1:
                    ax3.set_xticklabels([])

                ax3.tick_params(axis="x", labelsize=tick_size)
                ax3.tick_params(axis="y", labelsize=tick_size)

                ax3.spines[["top", "right"]].set_visible(False)

                if id_y_plot == 0 and id_x_plot == int(n_collapsed_groups / 2):
                    ax3.set_ylabel(
                        "Comp. {} average Δ burst rate (%)".format(comps_to_vis[1]),
                        fontsize=label_size,
                    )

                if id_x_plot == n_collapsed_groups - 1:
                    ax3.set_xlabel(
                        "Comp. {} average Δ burst rate (%)".format(comps_to_vis[0]),
                        fontsize=label_size,
                    )

            fig3.legend(
                frameon=False,
                title="Channel, condition",
                title_fontsize=legend_size,
                fontsize=legend_label_size,
                loc="upper right",
                alignment="left",
            )

            fig3.suptitle("Subject {}\nt={}s".format(subject, tp), fontsize=title_size)

            bag_of_figs3.append(
                join(savepath_gif, "{}_comp_comod_in_time_{}.png".format(band, t))
            )
            fig3.savefig(bag_of_figs3[t])
            plt.close()

    images = []
    gif_name = join(savepath_gif, "{}_band_comod.gif".format(band))
    for filename in bag_of_figs3:
        images.append(imageio.imread(filename))
    imageio.mimwrite(gif_name, images, fps=5)

    # Suptitles and legends.
    fig.suptitle("Subject {}".format(subject), fontsize=title_size)

    fig.legend(
        frameon=False,
        title="Channel, condition",
        title_fontsize=legend_size,
        fontsize=legend_label_size,
        loc="upper right",
        alignment="left",
    )

    if rate_computation == "independent":
        fig1.suptitle("Subject {}".format(subject), fontsize=title_size)
        fig2.suptitle("Subject {}".format(subject), fontsize=title_size)

        fig1.legend(
            frameon=False,
            title="Channel, condition",
            title_fontsize=legend_size,
            fontsize=legend_label_size,
            loc="upper right",
            alignment="left",
        )
        fig2.legend(
            frameon=False,
            title="Channel, condition",
            title_fontsize=legend_size,
            fontsize=legend_label_size,
            loc="upper right",
            alignment="left",
        )

    # Optional saving.
    if savefigs == True:
        figname = join(
            savepath,
            "sub_{}".format(subject),
            "{}_band_{}_delta_burst_rate{}.{}".format(
                band, rate_computation, fig_name_ind, plot_format
            ),
        )
        fig.savefig(figname, dpi=dpi, facecolor="w", edgecolor="w")

        if rate_computation == "independent":
            figname1 = join(
                savepath,
                "sub_{}".format(subject),
                "{}_band_{}_delta_burst_rate_plane{}.{}".format(
                    band, rate_computation, fig_name_ind, plot_format
                ),
            )
            fig1.savefig(figname1, dpi=dpi, facecolor="w", edgecolor="w")

            figname2 = join(
                savepath,
                "sub_{}".format(subject),
                "{}_band_{}_average_burst_rate{}.{}".format(
                    band, rate_computation, fig_name_ind, plot_format
                ),
            )
            fig2.savefig(figname2, dpi=dpi, facecolor="w", edgecolor="w")

    elif savefigs == False:
        plt.show()
