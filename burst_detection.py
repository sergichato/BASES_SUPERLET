"""
Iterative burst detection on single-channel time-frequency or lagged
coherence matrices algorithm.

Authors: James Bonaiuto <james.bonaiuto@isc.cnrs.fr>
         Maciej Szul <maciej.szul@isc.cnrs.fr>

Adaptation: Sotirios Papadopoulos <sotirios.papadopoulos@univ-lyon1.fr>
"""

import numpy as np
from mne.filter import filter_data
from scipy.signal import hilbert, argrelextrema
from scipy.stats import linregress

from burst_modeling import gaus2d, overlap, fwhm_burst_norm


# Burst extraction
def extract_bursts(
    raw_trials,
    TF,
    times,
    search_freqs,
    band_lims,
    fooof_thresh,
    sfreq,
    subject,
    channel,
    labels,
    w_size=0.26,
    remove_fooof=True,
    verbose=False,
):
    bursts = {
        "subject": [],
        "channel": [],
        "trial": [],
        "label": [],
        "waveform": [],
        "peak_freq": [],
        "peak_amp_iter": [],
        "peak_amp_base": [],
        "peak_time": [],
        "peak_adjustment": [],
        "fwhm_freq": [],
        "fwhm_time": [],
        "polarity": [],
        "volume": [],
    }

    # Compute ERF/ERP.
    erf = np.mean(raw_trials, axis=0)

    # Grid for computing 2D Gaussians
    x_idx, y_idx = np.meshgrid(range(len(times)), range(len(search_freqs)))

    # Window size in points
    wlen = int(w_size * sfreq)
    half_wlen = int(wlen * 0.5)

    # Iterate through trials
    for t_idx, tr in enumerate(TF):

        trial_bursts = {
            "subject": [],
            "channel": [],
            "trial": [],
            "label": [],
            "waveform": [],
            "peak_freq": [],
            "peak_amp_iter": [],
            "peak_amp_base": [],
            "peak_time": [],
            "peak_adjustment": [],
            "fwhm_freq": [],
            "fwhm_time": [],
            "polarity": [],
            "volume": [],
        }

        # Regress out ERF
        slope, intercept, _, _, _ = linregress(erf, raw_trials[t_idx, :])
        raw_trials[t_idx, :] = raw_trials[t_idx, :] - (intercept + slope * erf)
        
        # Optionally, subtract 1/f threshold.
        if remove_fooof == True:
            trial_TF = tr - fooof_thresh
            trial_TF[trial_TF < 0] = 0
        else:
            trial_TF = tr

        # Skip trial if all values are zero.
        # This should be the case for below-FOOOF threshold trials,
        # or below-sham lagged coherence threshold trials.
        if (trial_TF == 0).all():
            print(
                "\t\tSkipping trial {} because there don't exist any above-threshold values.".format(
                    t_idx
                )
            )
            continue

        # TF for iterating
        trial_TF_iter = np.copy(trial_TF)

        prnt_count = 0
        
        while True:
            
            # Compute noise floor.
            thresh = 2 * np.std(trial_TF_iter)

            # Find peak
            [peak_freq_idx, peak_time_idx] = np.unravel_index(
                np.argmax(trial_TF_iter), trial_TF.shape
            )
            peak_freq = search_freqs[peak_freq_idx]
            peak_amp_iter = trial_TF_iter[peak_freq_idx, peak_time_idx]
            peak_amp_base = trial_TF[peak_freq_idx, peak_time_idx]
            
            # Stop if no peak lies above the trheshold.
            if peak_amp_iter < thresh:
                if t_idx not in np.unique(bursts["trial"]):
                    if verbose == True:
                        print(
                            "\t\tTrial {} contains no more bursts in the frequency ".format(t_idx)
                            + "range of interest..."
                        )
                break

            # Fit 2D Gaussian and subtract from TF.
            if remove_fooof == True:
                right_loc, left_loc, up_loc, down_loc = fwhm_burst_norm(
                    trial_TF_iter, (peak_freq_idx, peak_time_idx)
                )
            elif remove_fooof == False:
                # Do not overestimate the fitted Gaussian.
                proxy_trial_TF_iter = np.copy(trial_TF_iter)
                proxy_trial_TF_iter = proxy_trial_TF_iter - 2 * np.std(
                    proxy_trial_TF_iter
                )
                proxy_trial_TF_iter[proxy_trial_TF_iter < 0] = 0
                right_loc, left_loc, up_loc, down_loc = fwhm_burst_norm(
                    proxy_trial_TF_iter, (peak_freq_idx, peak_time_idx)
                )

            # Detect and remove degenerate Gaussian (limits not found).
            vert_isnan = any(np.isnan([up_loc, down_loc]))
            horiz_isnan = any(np.isnan([right_loc, left_loc]))
            if vert_isnan:
                v_sh = int((search_freqs.shape[0] - peak_freq_idx) / 2)
                if v_sh <= 0:
                    v_sh = 1
                up_loc = v_sh
                down_loc = v_sh

            elif horiz_isnan:
                h_sh = int((times.shape[0] - peak_time_idx) / 2)
                if h_sh <= 0:
                    h_sh = 1
                right_loc = h_sh
                left_loc = h_sh

            hv_isnan = any([vert_isnan, horiz_isnan])

            # Compute FWHM and convert to SD.
            fwhm_f_idx = up_loc + down_loc
            fwhm_f = (search_freqs[1] - search_freqs[0]) * fwhm_f_idx
            fwhm_t_idx = left_loc + right_loc
            fwhm_t = (times[1] - times[0]) * fwhm_t_idx
            sigma_t = (fwhm_t_idx) / 2.355
            sigma_f = (fwhm_f_idx) / 2.355
            
            # Fitted Gaussian.
            z = peak_amp_iter * gaus2d(
                x_idx, y_idx, mx=peak_time_idx, my=peak_freq_idx, sx=sigma_t, sy=sigma_f
            )
            new_trial_TF_iter = trial_TF_iter - z

            # Proceed only if the fitted Gaussian is not degenerate
            # and within the requested frequency band limits.
            if all(
                [peak_freq >= band_lims[0], peak_freq <= band_lims[1], not hv_isnan]
            ):

                # Bandpass filtering within the burst frequency range.
                freq_range = [
                    np.max([0, peak_freq_idx - down_loc]),
                    np.min([len(search_freqs) - 1, peak_freq_idx + up_loc])
                ]
                filtered = filter_data(
                    raw_trials[t_idx, :].reshape(1, -1),
                    sfreq,
                    search_freqs[freq_range[0]],
                    search_freqs[freq_range[1]],
                    verbose=False
                )

                # Hilbert transform
                analytic_signal = hilbert(filtered)

                # Get phase
                instantaneous_phase = np.unwrap(np.angle(analytic_signal)) % np.pi

                # Find phase local minima (near 0)
                min_phase_pts = argrelextrema(instantaneous_phase.T, np.less)[0]

                new_peak_time_idx = peak_time_idx

                # Find local phase minima with negative deflection closest to TF peak.
                # If no minimum is found, the error is caught and no burst is added.
                try:
                    new_peak_time_idx = min_phase_pts[np.argmin(np.abs(peak_time_idx - min_phase_pts))]
                    adjustment = (new_peak_time_idx - peak_time_idx) * 1 / sfreq
                except:
                    if verbose == True:
                        if prnt_count == 0:
                            flm = "\t\tTrial {}:\n\t\t\tSkipping a candidate burst, ".format(
                                t_idx
                            )
                        else:
                            flm = "\t\t\tSkipping a candidate burst, "
                        print(
                            flm
                            + "as no local minima were detected in the instantaneous phase of the signal."
                        )
                    prnt_count += 1

                    adjustment = 1

                # Keep if adjustment less than 30ms.
                if np.abs(adjustment) < 0.03:
                    
                    # If burst won't be cutoff...
                    if (
                        new_peak_time_idx >= half_wlen
                        and new_peak_time_idx + half_wlen <= len(times)
                    ):
                        peak_time = times[new_peak_time_idx]

                        overlapped = False

                        # Check for overlap.
                        for b_idx in range(len(trial_bursts['peak_time'])):
                            
                            o_t = trial_bursts["peak_time"][b_idx]
                            o_fwhm_t = trial_bursts["fwhm_time"][b_idx]
                            
                            if overlap(
                                [peak_time - 0.5 * fwhm_t, peak_time + 0.5 * fwhm_t],
                                [o_t - 0.5 * o_fwhm_t, o_t + 0.5 * o_fwhm_t],
                            ):
                                overlapped = True
                                break

                        if not overlapped:
                            
                            # Get burst.
                            burst = raw_trials[
                                t_idx,
                                new_peak_time_idx
                                - half_wlen : new_peak_time_idx
                                + half_wlen,
                            ]

                            # Remove DC offset.
                            burst = burst - np.mean(burst)
                            burst_times = (
                                times[
                                    new_peak_time_idx
                                    - half_wlen : new_peak_time_idx
                                    + half_wlen
                                ]
                                - times[new_peak_time_idx]
                            )

                            # Flip if positive deflection.
                            peak_dists = np.abs(
                                argrelextrema(filtered.T, np.greater)[0] - new_peak_time_idx
                            )
                            trough_dists = np.abs(
                                argrelextrema(filtered.T, np.less)[0] - new_peak_time_idx
                            )

                            polarity = 0
                            if len(trough_dists) == 0 or (
                                len(peak_dists) > 0
                                and np.min(peak_dists) < np.min(trough_dists)
                            ):
                                burst *= -1.0
                                polarity = 1

                            trial_bursts["trial"].append(int(t_idx))
                            trial_bursts["subject"].append(subject)
                            trial_bursts["channel"].append(channel)
                            trial_bursts["label"].append(labels[t_idx])
                            trial_bursts["waveform"].append(burst)
                            trial_bursts["peak_freq"].append(peak_freq)
                            trial_bursts["peak_amp_iter"].append(peak_amp_iter)
                            trial_bursts["peak_amp_base"].append(peak_amp_base)
                            trial_bursts["peak_time"].append(peak_time)
                            trial_bursts["peak_adjustment"].append(adjustment)
                            trial_bursts["fwhm_freq"].append(fwhm_f)
                            trial_bursts["fwhm_time"].append(fwhm_t)
                            trial_bursts["polarity"].append(polarity)
                            trial_bursts["volume"].append(np.sum(z))

            trial_TF_iter = new_trial_TF_iter
        
        bursts["trial"].extend(trial_bursts["trial"])
        bursts["subject"].extend(trial_bursts["subject"])
        bursts["channel"].extend(trial_bursts["channel"])
        bursts["label"].extend(trial_bursts["label"])
        bursts["waveform"].extend(trial_bursts["waveform"])
        bursts["peak_freq"].extend(trial_bursts["peak_freq"])
        bursts["peak_amp_iter"].extend(trial_bursts["peak_amp_iter"])
        bursts["peak_amp_base"].extend(trial_bursts["peak_amp_base"])
        bursts["peak_time"].extend(trial_bursts["peak_time"])
        bursts["peak_adjustment"].extend(trial_bursts["peak_adjustment"])
        bursts["fwhm_freq"].extend(trial_bursts["fwhm_freq"])
        bursts["fwhm_time"].extend(trial_bursts["fwhm_time"])
        bursts["polarity"].extend(trial_bursts["polarity"])
        bursts["volume"].extend(trial_bursts["volume"])

    bursts["subject"] = np.array(bursts["subject"])
    bursts["channel"] = np.array(bursts["channel"])
    bursts["trial"] = np.array(bursts["trial"])
    bursts["label"] = np.array(bursts["label"])
    bursts["waveform"] = np.array(bursts["waveform"])
    try:
        bursts["waveform_times"] = burst_times
    except:
        bursts["waveform_times"] = []
    bursts["peak_freq"] = np.array(bursts["peak_freq"])
    bursts["peak_amp_iter"] = np.array(bursts["peak_amp_iter"])
    bursts["peak_amp_base"] = np.array(bursts["peak_amp_base"])
    bursts["peak_time"] = np.array(bursts["peak_time"])
    bursts["peak_adjustment"] = np.array(bursts["peak_adjustment"])
    bursts["fwhm_freq"] = np.array(bursts["fwhm_freq"])
    bursts["fwhm_time"] = np.array(bursts["fwhm_time"])
    bursts["polarity"] = np.array(bursts["polarity"])
    bursts["volume"] = np.array(bursts["volume"])

    return bursts
