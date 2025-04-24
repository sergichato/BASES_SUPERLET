"""
Functions that compute burst rate and statistically significant clusters
of burst rate in a time-resolved manner.
"""

import numpy as np
from os.path import join

from mne.stats import permutation_cluster_test
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from help_funcs import significant_features


def compute_power(
    subject,
    epochs,
    labels,
    exp_variables,
    tf_method,
    band,
    channel_ids,
    zapit,
    noise_freq,
    noise_wins,
    savepath,
    remove_fooof=True,
    smooth=True,
):
    """
    Beta band power computation per trial and channel for a given condition.

    Parameters
    ----------
    subject: int
             Subject to be analyzed.
    epochs: MNE-python epochs object
            Epochs object corresponding to a specific dataset.
    labels: list
            Labels corresponding to 'epochs'.
    exp_variables: dict
                   Experimental variables contained in the corresponding
                   'variables.json' file.
    tf_method: str {"wavelets", "superlets"}
               String indicating the algorithm used for burst
               extraction.
    band: str {"mu", "beta", "mu_beta"}
          Select band for burst detection.
    channel_ids: list
                 Indices of channels to keep for the analysis.
    zapit: bool
           If set to "True", iteratively remove a noise artifact from the raw
           signal. The frequency of the artifact is provided by 'this_freq'.
    noise_freq: int or None
               When set to "int", frequency containing power line noise, or
               equivalent artifact. Only considered if 'zapit' is "True".
    noise_wins: list or None
                Window sizes for removing line noise.  Only considered if
                'zapit' is "True".
    savepath: str
              Parent directory that contains all results. Defaults
              to the path provided in the corresponding 'variables.json'
              file.
    remove_fooof: bool, optional
                  Removed aperiodic FOOOF spectrum fit from time-frequency
                  matrices.
                  Defaults to "True".
    smooth: bool, optional
            If "True", apply smoothing.
            Defaults to "True".

    Returns
    -------
    epochs_xx: numpy array
               Array containing the trial-averaged, baseline-corrected
               Hilbert envelope power for a given condition.
    """

    # Time period of task.
    tmin = exp_variables["tmin"]
    tmax = exp_variables["tmax"]
    exp_time_periods = exp_variables["exp_time_periods"]
    sfreq = exp_variables["sfreq"]

    exp_time = np.linspace(tmin, tmax, int((np.abs(tmin) + np.abs(tmax)) * sfreq) + 1)

    baseline_time_lims = [exp_time_periods[0], exp_time_periods[1]]

    # Binned time axis.
    baseline_begin = int(np.where(exp_time == exp_time_periods[0])[0])
    rebound_end = int(np.where(exp_time == exp_time_periods[3])[0])
    bin_dt = 0.10
    erds_time = exp_time[baseline_begin : rebound_end + 1]
    binning = np.arange(erds_time[0], erds_time[-1] + bin_dt, bin_dt)
    binned_exp_time = np.around(binning, decimals=2)
    binned_exp_time = binned_exp_time[3:-3]
    baseline_bins = np.where(
        (binned_exp_time >= baseline_time_lims[0])
        & (binned_exp_time <= baseline_time_lims[1])
    )[0]

    # Labels.
    unique_labels = np.unique(labels)
    labels_1 = np.where(labels == unique_labels[0])[0]
    labels_2 = np.where(labels == unique_labels[1])[0]

    # Suject-specific directory.
    sub_dir = join(savepath, "sub_{}/".format(subject))

    # Power band.
    channels = exp_variables["channels"]

    # Frequency bands of interest.
    if remove_fooof == False:
        if band == "beta":
            power_band = np.tile([15, 30], (len(channels), 1))
        elif band == "mu":
            power_band = np.tile([8, 15], (len(channels), 1))
        elif band == "mu_beta":
            power_band = np.tile([8, 30], (len(channels, 1)))
    
    elif remove_fooof == True:
        if band == "beta":
            custom_betas = np.load(
                join(sub_dir, "beta_bands_{}.npy".format(tf_method))
            )
            for j, custom_beta in enumerate(custom_betas):
                # Ascertain valid ranges.
                if np.isnan(custom_beta[0]):
                    custom_betas[j][0] = 15
                if np.isnan(custom_beta[1]):
                    custom_betas[j][1] = 30
            power_band = custom_betas
        
        elif band == "mu":
                custom_mus = np.load(join(sub_dir, "mu_bands_{}.npy".format(tf_method)))
                for j, custom_mu in enumerate(custom_mus):
                    # Ascertain valid ranges.
                    if np.isnan(custom_mu[0]):
                        custom_mus[j][0] = 8
                    if np.isnan(custom_mu[1]):
                        custom_mus[j][1] = 13
                power_band = custom_mus
        
        elif band == "mu_beta":
            custom_betas = np.load(
                join(sub_dir, "beta_bands_{}.npy".format(tf_method))
            )
            custom_mus = np.load(join(sub_dir, "mu_bands_{}.npy".format(tf_method)))

            power_band = []
            for custom_beta, custom_mu in zip(custom_betas, custom_mus):
                custom_band = np.array(
                    [np.nanmin(custom_mu), np.nanmax(custom_beta)]
                )
                # Ascertain valid ranges.
                if np.isnan(custom_band[0]):
                    custom_band[0] = 8
                if np.isnan(custom_band[1]):
                    custom_band[1] = 30
                power_band.append(custom_band)

    # Filtering with channel-specific custom frequency band.
    for cb, c in zip(power_band, channel_ids):
        epochs.filter(cb[0], cb[1], picks=channels[c])

    # Power.
    epochs.apply_hilbert(envelope=True)
    epochs = epochs.get_data()
    epochs = epochs**2

    # Trim time to match task duration and keep selected channels.
    epochs = epochs[:, channel_ids, :]

    # Binning.
    epochs_power = np.zeros(
        [epochs.shape[0], epochs.shape[1], len(binned_exp_time) - 1]
    )
    for b, bin in enumerate(binned_exp_time[:-1]):
        window = np.where((exp_time >= bin) & (exp_time < binned_exp_time[b + 1]))[0]
        epochs_power[:, :, b] = np.sum(epochs[:, :, window], axis=-1)

    if smooth == True:
        # Kernel std.
        kernel_std = 3

        for tr in range(epochs_power.shape[0]):
            epochs_power[tr, :, :] = gaussian_filter1d(
                epochs_power[tr, :, :], kernel_std
            )

    # Baseline correction.
    epochs_1 = epochs_power[labels_1, :, :]
    epochs_2 = epochs_power[labels_2, :, :]

    epochs_11 = epochs_1[:, 0, :]
    epochs_12 = epochs_1[:, 1, :]
    epochs_21 = epochs_2[:, 0, :]
    epochs_22 = epochs_2[:, 1, :]

    base_11 = np.mean(epochs_11[:, baseline_bins], axis=(0, 1)).reshape(-1, 1)
    base_12 = np.mean(epochs_12[:, baseline_bins], axis=(0, 1)).reshape(-1, 1)
    base_21 = np.mean(epochs_21[:, baseline_bins], axis=(0, 1)).reshape(-1, 1)
    base_22 = np.mean(epochs_22[:, baseline_bins], axis=(0, 1)).reshape(-1, 1)

    epochs_11 = (epochs_11 - base_11) / base_11 * 100
    epochs_12 = (epochs_12 - base_12) / base_12 * 100
    epochs_21 = (epochs_21 - base_21) / base_21 * 100
    epochs_22 = (epochs_22 - base_22) / base_22 * 100

    epochs_11_sem = np.std(epochs_11, axis=0) / np.sqrt(epochs_11.shape[0])
    epochs_12_sem = np.std(epochs_12, axis=0) / np.sqrt(epochs_12.shape[0])
    epochs_21_sem = np.std(epochs_21, axis=0) / np.sqrt(epochs_21.shape[0])
    epochs_22_sem = np.std(epochs_22, axis=0) / np.sqrt(epochs_22.shape[0])

    epochs_11 = np.mean(epochs_11, axis=0)
    epochs_12 = np.mean(epochs_12, axis=0)
    epochs_21 = np.mean(epochs_21, axis=0)
    epochs_22 = np.mean(epochs_22, axis=0)

    epochs_dict = {
        "epochs_11": epochs_11,
        "epochs_11_sem": epochs_11_sem,
        "epochs_12": epochs_12,
        "epochs_12_sem": epochs_12_sem,
        "epochs_21": epochs_21,
        "epochs_21_sem": epochs_21_sem,
        "epochs_22": epochs_22,
        "epochs_22_sem": epochs_22_sem,
    }

    return epochs_dict


def compute_burst_rate(
    subject_dictionary,
    trials_cond,
    channel_ids,
    binned_plot_time,
    baseline_bins,
    bin_dt,
):
    """
    Burst rate computation per trial and channel for a given condition.

    Parameters
    ----------
    subject_dictionary: dict
                        Dictionary containing all detected bursts of a subject
                        for a given condition.
    trials_cond: numpy array
                 Indices of trials corresponding to a given condition.
    channel_ids: list
                 Indices of channels to take into account.
    binned_plot_time: numpy array
                      Array representing the trimmed experimental time
                      with wide time steps, needed for an estimation
                      of burst rate in non-overlapping windows.
    baseline_bins: numpy array
                   Bins corresponding to the baseline period.
    bin_dt: float
            Time step for advancing to a new time window.

    Returns
    -------
    burst_rate_1/2: numpy array
                    Array containing the trial-averaged, baseline-corrected
                    burst rate for a given condition.
    """

    # Kernel std.
    kernel_std = 3

    # Only consider bursts above the 75th amplitude quartile.
    lab = np.where(
        subject_dictionary["peak_amp_iter"]
        >= np.percentile(subject_dictionary["peak_amp_iter"], 75)
    )[0]

    burst_rate_1 = np.zeros((len(trials_cond), len(binned_plot_time) - 1))
    burst_rate_2 = np.zeros((len(trials_cond), len(binned_plot_time) - 1))

    # For each trial...
    for tr, trial in enumerate(trials_cond):

        ids_1 = np.where(
            (subject_dictionary["trial"][lab] == trial)
            & (subject_dictionary["channel"][lab] == channel_ids[0])
        )[0]
        ids_2 = np.where(
            (subject_dictionary["trial"][lab] == trial)
            & (subject_dictionary["channel"][lab] == channel_ids[1])
        )[0]

        mv_hist_1, _ = np.histogram(
            subject_dictionary["peak_time"][lab][ids_1], bins=binned_plot_time
        )
        mv_hist_2, _ = np.histogram(
            subject_dictionary["peak_time"][lab][ids_2], bins=binned_plot_time
        )

        # Compute burst rate.
        burst_rate_1[tr, :] = gaussian_filter1d((mv_hist_1 / bin_dt), kernel_std)
        burst_rate_2[tr, :] = gaussian_filter1d((mv_hist_2 / bin_dt), kernel_std)

    # Baseline correction.
    base_1 = np.mean(burst_rate_1[:, baseline_bins], axis=(0, 1)).reshape(-1, 1)
    base_2 = np.mean(burst_rate_2[:, baseline_bins], axis=(0, 1)).reshape(-1, 1)

    burst_rate_1 = (burst_rate_1 - base_1) / base_1 * 100
    burst_rate_2 = (burst_rate_2 - base_2) / base_2 * 100

    burst_rate_1_sem = np.std(burst_rate_1, axis=0) / np.sqrt(burst_rate_1.shape[0])
    burst_rate_2_sem = np.std(burst_rate_2, axis=0) / np.sqrt(burst_rate_2.shape[0])

    burst_rate_1 = np.mean(burst_rate_1, axis=0)
    burst_rate_2 = np.mean(burst_rate_2, axis=0)

    return burst_rate_1, burst_rate_1_sem, burst_rate_2, burst_rate_2_sem


def compute_burst_rate_nd(
    subject_dictionary,
    trials_cond,
    channel_ids,
    scores_cond,
    measure_bins,
    binned_plot_time,
    bin_dt,
    n_dims=2,
    data_type="single",
):
    """
    Burst rate computation per trial and channel for a given condition.

    Parameters
    ----------
    subject_dictionary: dict
                        Dictionary containing all detected bursts of a subject.
    trials_cond: numpy array
                 Indices of trials corresponding to a given condition.
    channel_ids: list
                 Indices of channels to take into account.
    sub_scores_dists: numpy array
                      Array containing all burst scores along a given component
                      axis (based on all trials) for a subject.
    measure_bins: numpy array
                  Bins of the scores axis.
    binned_plot_time: numpy array
                      Array representing the trimmed experimental time
                      with wide time steps, needed for an estimation
                      of burst rate in non-overlapping windows.
    bin_dt: float
            Time step for advancing to a new time window.
    data_type: str {"single", "independent", "joint"}, optional
               String controlling the dimensions of the returned variables.
               If "single" return a 3D tensor, else a 4D tensor with different
               final dimension for each case.
               Defaults to "single".

    Returns
    -------
    mv_burst_rate_cond_1, mv_burst_rate_cond_2: numpy array
                                                Burst rate per channel.
    mv_burst_rate_common: numpy array
                          Burst rate across conditions.
    """
    # Check validity of default values.
    if data_type == "single" or data_type == "independent" or data_type == "joint":
        pass
    else:
        raise ValueError(
            "'data_type' must be one of 'single', 'independent' or 'joint'."
        )

    # Smoothing kernels' standard deviations and burst rate variables' initialization.
    if data_type == "single" or data_type == "independent":
        kernel_stds = [3, 6]

        mv_burst_rate_cond_1 = np.zeros(
            (len(trials_cond), len(binned_plot_time) - 1, len(measure_bins) - 1)
        )
        mv_burst_rate_cond_2 = np.zeros(
            (len(trials_cond), len(binned_plot_time) - 1, len(measure_bins) - 1)
        )

    elif data_type == "joint":
        kernel_stds = [3, 6, 6]

        mv_burst_rate_cond_1 = np.zeros(
            (
                len(trials_cond),
                len(binned_plot_time) - 1,
                len(measure_bins[1]) - 1,
                len(measure_bins[2]) - 1,
            )
        )
        mv_burst_rate_cond_2 = np.zeros(
            (
                len(trials_cond),
                len(binned_plot_time) - 1,
                len(measure_bins[1]) - 1,
                len(measure_bins[2]) - 1,
            )
        )

    if data_type == "single":
        mv_burst_rate_common = np.zeros(
            (2 * len(trials_cond), len(binned_plot_time) - 1, len(measure_bins) - 1)
        )

    for t_idx, trial_cond in enumerate(trials_cond):

        t_bursts = np.where(subject_dictionary["trial"] == trial_cond)[0]
        c_bursts_1 = np.where(
            subject_dictionary["channel"][t_bursts] == channel_ids[0]
        )[0]
        c_bursts_2 = np.where(
            subject_dictionary["channel"][t_bursts] == channel_ids[1]
        )[0]

        # Smoothed count per ROI.
        if data_type != "joint":
            mv_hist_cond_1, _, _ = np.histogram2d(
                subject_dictionary["peak_time"][t_bursts][c_bursts_1],
                scores_cond[t_bursts][c_bursts_1],
                bins=[binned_plot_time, measure_bins],
            )
            mv_hist_cond_2, _, _ = np.histogram2d(
                subject_dictionary["peak_time"][t_bursts][c_bursts_2],
                scores_cond[t_bursts][c_bursts_2],
                bins=[binned_plot_time, measure_bins],
            )

            # Across conditions.
            if data_type == "single":
                mv_burst_rate_common[t_idx, :, :] = gaussian_filter(
                    mv_hist_cond_1, kernel_stds
                )
                mv_burst_rate_common[len(trials_cond) + t_idx, :, :] = gaussian_filter(
                    mv_hist_cond_2, kernel_stds
                )

        else:
            mv_hist_cond_1, _ = np.histogramdd(
                np.vstack(
                    (
                        subject_dictionary["peak_time"][t_bursts][c_bursts_1],
                        scores_cond[:, t_bursts][:, c_bursts_1],
                    )
                ).T,
                bins=measure_bins,
            )

            mv_hist_cond_2, _ = np.histogramdd(
                np.vstack(
                    (
                        subject_dictionary["peak_time"][t_bursts][c_bursts_2],
                        scores_cond[:, t_bursts][:, c_bursts_2],
                    )
                ).T,
                bins=measure_bins,
            )

        # Count to rate.
        mv_hist_cond_1 = mv_hist_cond_1 / bin_dt
        mv_hist_cond_2 = mv_hist_cond_2 / bin_dt

        # Smoothing.
        mv_burst_rate_cond_1[t_idx, :] = gaussian_filter(mv_hist_cond_1, kernel_stds)        
        mv_burst_rate_cond_2[t_idx, :] = gaussian_filter(mv_hist_cond_2, kernel_stds)

    if data_type == "single":
        return mv_burst_rate_cond_1, mv_burst_rate_cond_2, mv_burst_rate_common
    else:
        return mv_burst_rate_cond_1, mv_burst_rate_cond_2


def significant_components(
    sub_dict_cond1,
    sub_dict_cond2,
    sub_dict_trials_cond1,
    sub_dict_trials_cond2,
    channels,
    sub_scores_dists,
    comps_to_analyze,
    comps_groups,
    binned_time,
    bin_dt,
    task_time_lims,
    significance_threshold=0.25,
):
    """
    Statistically significant clusters of burst modulation across the time and score dimensions for
    two channels (most interestingly C3 and C4) per component and experimental condition.

    Parameters
    ----------
    sub_dict_cond1: dict
                    Dictionary containing all detected bursts of 'subject' for condition 1.
    sub_dict_cond2 : dict
                    Dictionary containing all detected bursts of 'subject' for condition 2.
    sub_dict_trials_cond1: list
                           Indices of "condition 1" trials.
    sub_dict_trials_cond2: list
                           Indices of "condition 2" trials.
    channels: list
              Names of channels used while creation burst features. Search for C3 and C4 only.
    sub_scores_dists: list
                      List containing all burst scores along each component axis
                      (based on all trials) for 'subject'.
    comps_to_analyze: list or numpy array
                      List of the indices of components used.
    comps_groups: int
                  Number of groups for the scores of each component axis should be split into.
    binned_time: numpy array
                 Array representing the trimmed experimental time with wide time steps,
                 needed for an estimation of burst rate in non-overlapping windows.
    bin_dt: float
            Time step for advancing to a new time window.
    significance_threshold: float, optional
                        Fraction of pixels that indicates the lower bound for marking a
                        feature as significant. Ignored if 'show_stats' is set to "False".
                        Defaults to 0.25.

    Returns
    -------
    gtk: list
         List filled iteratively with the indices of the features that overlap
         with statistically significant clusters along each component axis.
         Look at 'significant_features' function.
    """

    # Indices of channels of interest.
    channel_ids = [
        np.where(np.array(channels) == "C3")[0],
        np.where(np.array(channels) == "C4")[0],
    ]

    # Count number of bursts in each time window and measure bin.
    trials_cond1 = np.unique(sub_dict_cond1["trial"])
    trials_cond2 = np.unique(sub_dict_cond2["trial"])

    # Trim time axis to exclude edge effects and define baseline.
    binned_plot_time = binned_time[3:-3]
    task_bins = np.where(
        (binned_time >= task_time_lims[0]) & (binned_time <= task_time_lims[1])
    )[0]

    # Binning of burst scores axis.
    score_bins = 41

    # Initialization of group feature ids with significant overlap
    # with clusters of burst probability modulation.
    feature_group_ids_cond1 = []
    feature_group_ids_cond2 = []
    gtk = []

    # Iteratation over components.
    for comp_name in comps_to_analyze:

        # Common scores limits for all subjects.
        measure_bins = np.linspace(
            np.min(sub_scores_dists[comp_name - 1]),
            np.max(sub_scores_dists[comp_name - 1]),
            score_bins,
        )
        scores_cond1 = sub_scores_dists[comp_name - 1][np.hstack(sub_dict_trials_cond1)]
        scores_cond2 = sub_scores_dists[comp_name - 1][np.hstack(sub_dict_trials_cond2)]

        # Limits in the metrics that are used to split each image in features.
        if comps_groups == 1:
            raise ValueError(
                "You need to specify at least 2 groups when creating burst features!"
            )
        else:
            iqrs = np.linspace(
                np.min(sub_scores_dists[comp_name - 1]),
                np.max(sub_scores_dists[comp_name - 1]),
                comps_groups + 1,
            )
            scores_lims = []
            for i in range(comps_groups):
                scores_lims.append([iqrs[i], iqrs[i + 1]])
            scores_lims = np.array(scores_lims)

        # Subject burst rates.
        mv_burst_rate_cond1_1, mv_burst_rate_cond1_2, _ = compute_burst_rate_2D(
            sub_dict_cond1,
            trials_cond1,
            channel_ids,
            scores_cond1,
            measure_bins,
            binned_plot_time,
            bin_dt,
        )
        mv_burst_rate_cond2_1, mv_burst_rate_cond2_2, _ = compute_burst_rate_2D(
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

        # Evaluate the overlap of significant clusters with each feature group.
        significant_features(
            F_obs_1,
            scores_lims,
            score_bins,
            task_bins,
            feature_group_ids_cond1,
            comp_name - 1,
            comps_groups,
            significance_threshold=significance_threshold,
        )
        significant_features(
            F_obs_2,
            scores_lims,
            score_bins,
            task_bins,
            feature_group_ids_cond2,
            comp_name - 1,
            comps_groups,
            significance_threshold=significance_threshold,
        )
        significant_features(
            F_obs_3,
            scores_lims,
            score_bins,
            task_bins,
            gtk,
            comp_name - 1,
            comps_groups,
            significance_threshold=significance_threshold,
        )

    return gtk
