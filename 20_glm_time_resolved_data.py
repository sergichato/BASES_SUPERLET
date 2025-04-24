import numpy as np
import pandas as pd
from os.path import join

from moabb.datasets import (
    Zhou2016,
    BNCI2014004,
    BNCI2014001,
    MunichMI,
    Weibo2014,
    Cho2017,
)

from preprocess import (
    load_sub,
    apply_preprocessing,
    Dreyer2023,
)
from help_funcs import load_exp_variables


# ----- #
# Hyperparameters.
remove_fooof = False
if remove_fooof == True:
    fooof_save_str = ""
elif remove_fooof == False:
    fooof_save_str = "_nfs"

metric = "rocauc"
if metric == "rocauc":
    metric_str = "score"
elif metric == "accuracy":
    metric_str = metric

acc_time_index = False  # False, True

classification_mode = "incremental" # "incremental", "sliding"

if classification_mode == "incremental":
    clm_str = "_tr"
elif classification_mode == "sliding":
    clm_str = "_sl"

sl_time_center = "start"  # "start", "mid", "end"


# ----- #
def itr_tr(score, time, classification_mode, bin_dt):

    # Binary entropy.
    H = -score * np.log2(score) - (1 - score) * np.log2(1 - score)

    # A perfect score of 1 results in NAN, so set it to 0.
    H[np.where(np.isnan(H))] = 0

    # Time shift to only use positive values.
    if classification_mode == "incremental":
        new_time = [t - time[0] + bin_dt if t <= 0 else t for t in time]
        new_time = np.around(new_time, 3)
    elif classification_mode == "sliding":
        new_time = time - time[0] + bin_dt

    # ITR.
    itr = (1 - H) / new_time

    return itr, new_time


# ----- #
# Dataset selection.
datas = [
    "zhou2016",
    "2014004",
    "2014001",
    "weibo2014",
    "munichmi",
    "cho2017",
    "dreyer2023",
]

# Mode.
mode = "local"  # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/"

res_to_load = [
    "conv",
    "fb_1_beta",
    "fb_1_mubeta",
    "fb_beta",
    "fb_mubeta",
]

filter_banks = [
    [[6, 9], [9, 12], [12, 15]],
    [[15, 18], [18, 21], [21, 24], [24, 27], [27, 30]],
    [[6, 9], [9, 12], [12, 15], [15, 18], [18, 21], [21, 24], [24, 27], [27, 30]],
    [[6, 15]],
    [[15, 30]],
    [[6, 30]],
]


# ----- #
for j in range(len(res_to_load)):

    # ----- #
    # Data parsing.
    all_datasets = []
    all_subjects = []
    all_variables = []
    all_results = []
    all_times = []
    all_times90 = []
    all_trials = []

    for d, data in enumerate(datas):
        if data == "zhou2016":
            dataset = Zhou2016()
            variables_path = "{}zhou_2016/variables.json".format(basepath)
            title_str = "Zhou 2016"
        elif data == "2014004":
            dataset = BNCI2014004()
            variables_path = "{}2014_004/variables.json".format(basepath)
            title_str = "BNCI 2014-004"
        elif data == "2014001":
            dataset = BNCI2014001()
            variables_path = "{}2014_001/variables.json".format(basepath)
            title_str = "BNCI 2014-001"
        elif data == "munichmi":
            dataset = MunichMI()
            variables_path = "{}munichmi/variables.json".format(basepath)
            title_str = "Munich MI (Grosse-Wentrup)"
        elif data == "cho2017":
            dataset = Cho2017()
            variables_path = "{}cho_2017/variables.json".format(basepath)
            title_str = "Cho 2017"
        elif data == "weibo2014":
            dataset = Weibo2014()
            variables_path = "{}weibo_2014/variables.json".format(basepath)
            title_str = "Weibo 2014"
        elif data == "dreyer2023":
            dataset = Dreyer2023(basepath=basepath + "dreyer_2023/")
            variables_path = "{}dreyer_2023/variables.json".format(basepath)
            title_str = "Dreyer 2023"

        # ----- #
        # Loading of dataset-specific variables.
        experimental_vars = load_exp_variables(json_filename=variables_path)

        savepath = experimental_vars["dataset_path"]

        subs = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
        if data == "cho2017":
            # Some subjects are not included in the dataset.
            subs = np.delete(np.array(subs), [31, 45, 48]).tolist()

        # Time-related variables.
        exp_time_periods = experimental_vars["exp_time_periods"]

        bin_dt = experimental_vars["bin_dt"]
        tmin = experimental_vars["tmin"]
        tmax = experimental_vars["tmax"]
        sfreq = experimental_vars["sfreq"]

        # Time in experiment.
        exp_time = np.linspace(tmin, tmax, int((np.abs(tmax - tmin)) * sfreq) + 1)
        exp_time = np.around(exp_time, decimals=3)
        task_time_lims = [exp_time_periods[1], exp_time_periods[2]]
        base_start = -0.5

        samples_step = np.around(exp_time[1] - exp_time[0], decimals=4)
        window_samples = int(bin_dt / samples_step)

        baseline_window_end = np.where(exp_time <= task_time_lims[0] - 0.5)[0][-1]
        rebound_window_end = np.where(exp_time <= task_time_lims[1] + 0.5)[0][-1]
        n_windows = int((rebound_window_end - baseline_window_end) / window_samples) + 1

        window_length_sl = 1.0
        step_length = 0.05
        if classification_mode == "sliding":
            bin_dt = step_length
        window_samples_sl = window_length_sl / samples_step
        step_samples = step_length / samples_step
        n_windows_sl = int(
            np.floor(
                ((rebound_window_end - baseline_window_end) - window_samples_sl)
                / step_samples
            )
            + window_samples_sl / step_samples
            + 1
        )

        # Time vectors.
        # Time in 100ms non-overlapping windows.
        time = np.array(
            [
                exp_time[baseline_window_end + n * window_samples]
                for n in range(n_windows)
            ]
        )

        # Time in 1000ms sliding (by 50ms) windows.
        time_sl = [
            exp_time[int(n * step_samples + step_samples)]
            for n in range(n_windows_sl)
        ]
        time_sl = np.array(time_sl)

        if sl_time_center == "mid":
            time_sl = [t + 0.5 for t in time_sl]
            time_sl = np.around(time_sl, 3)
        elif sl_time_center == "end":
            time_sl = [t + 1.0 for t in time_sl]
            time_sl = np.around(time_sl, 3)
        
        # Shifted ITR time of sliding window technique.
        new_time_sl = time_sl - time_sl[0] + bin_dt

        # "Important" time points.
        time_points = [0.0, exp_time_periods[2]]
        if data != "cho2017" and data != "dreyer2023":
            time_ids = [np.where(time == tp)[0][0] for tp in time_points]
            time_ids_sl = [np.where(time_sl == tp)[0][0] for tp in time_points]
        else:
            time_ids = [np.where(time >= tp)[0][0] for tp in time_points]
            time_ids_sl = [np.where(time_sl >= tp)[0][0] for tp in time_points]

        # ----- #
        # Dataset id.
        all_datasets.append(np.repeat(d + 1, len(subs) * len(res_to_load)))

        # Subject id.
        all_subjects.append(
            np.hstack(
                np.hstack(
                    [np.repeat(v + 1, len(res_to_load)) for v in range(len(subs))]
                )
            )
        )

        # Feature id.
        all_variables.append(
            np.tile([v + 1 for v in range(len(res_to_load))], len(subs))
        )

        # ----- #
        # Loading of decoding results.
        dataset_results = []

        if "conv" in res_to_load:
            dataset_results.append(
                np.mean(
                    np.load(
                        savepath
                        + "mean_{}_power_beta_band_csp{}_conv_waves{}.npy".format(
                            metric, fooof_save_str, clm_str
                        )
                    ),
                    axis=(2, 3),
                ),
            )

        if "fb_beta" in res_to_load:
            dataset_results.append(
                np.mean(
                    np.load(
                        savepath
                        + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                            metric,
                            len(filter_banks[1]),
                            filter_banks[1][0][0],
                            filter_banks[1][-1][1],
                            clm_str,
                        )
                    ),
                    axis=(2, 3),
                ),
            )

        if "fb_mu" in res_to_load:
            dataset_results.append(
                np.mean(
                    np.load(
                        savepath
                        + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                            metric,
                            len(filter_banks[0]),
                            filter_banks[0][0][0],
                            filter_banks[0][-1][1],
                            clm_str,
                        )
                    ),
                    axis=(2, 3),
                ),
            )

        if "fb_mubeta" in res_to_load:
            dataset_results.append(
                np.mean(
                    np.load(
                        savepath
                        + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                            metric,
                            len(filter_banks[2]),
                            filter_banks[2][0][0],
                            filter_banks[2][-1][1],
                            clm_str,
                        )
                    ),
                    axis=(2, 3),
                ),
            )

        if "fb_1_mu" in res_to_load:
            dataset_results.append(
                np.mean(
                    np.load(
                        savepath
                        + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                            metric,
                            len(filter_banks[3]),
                            filter_banks[3][0][0],
                            filter_banks[3][-1][1],
                            clm_str,
                        )
                    ),
                    axis=(2, 3),
                ),
            )

        if "fb_1_beta" in res_to_load:
            dataset_results.append(
                np.mean(
                    np.load(
                        savepath
                        + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                            metric,
                            len(filter_banks[4]),
                            filter_banks[4][0][0],
                            filter_banks[4][-1][1],
                            clm_str,
                        )
                    ),
                    axis=(2, 3),
                ),
            )

        if "fb_1_mubeta" in res_to_load:
            dataset_results.append(
                np.mean(
                    np.load(
                        savepath
                        + "mean_{}_power_filter_bank_{}_csp_{}_{}{}.npy".format(
                            metric,
                            len(filter_banks[5]),
                            filter_banks[5][0][0],
                            filter_banks[5][-1][1],
                            clm_str,
                        )
                    ),
                    axis=(2, 3),
                ),
            )

        # All subjects' results.
        n_trials = []
        sub_results = []
        sub_times = []
        sub_times90 = []

        for s, subj in enumerate(subs):

            # ----- #
            # Load number of trials.
            if data != "dreyer2023":
                sub_dir = join(savepath, "sub_{}/".format(subj))
                trials = np.load(join(sub_dir, "ntrials_all.npy"))
                n_trials.append(np.sum(trials[0][2:]))
            elif data == "dreyer2023":
                trials = np.load(join(savepath, "ntrials_all.npy"))
                n_trials.append(np.sum(trials[s, 2:]))

            # ----- #
            # Corresponding max classification results and time to this score,
            # only consideting time points within the trial period.
            if acc_time_index == False:
                sub_results.append(
                    [
                        np.max(dataset_result[s, time_ids[0] + 1 : time_ids[1]])
                        if classification_mode == "incremental"
                        else np.max(dataset_result[s, 4:])
                        for dataset_result in dataset_results
                    ]
                )
                sub_times.append(
                    [
                        time[
                            np.argmax(
                                dataset_result[s, time_ids[0] + 1 : time_ids[1]]
                            )
                        ]
                        if classification_mode == "incremental"
                        else new_time_sl[4:][
                            np.argmax(
                                dataset_result[
                                    s, 4:
                                ]
                            )
                        ]
                        for dataset_result in dataset_results
                    ]
                )

            elif acc_time_index == True:

                # Exclude first time points from analysis, because of
                # high ITR artifact due to very short decoding windows.
                itrs = [
                    itr_tr(
                        (
                            dataset_result[s, time_ids[0] + 1 : time_ids[1]]
                            if classification_mode == "incremental"
                            else dataset_result[s, 4:]
                        ),
                        (
                            time[time_ids[0] + 1 : time_ids[1]]
                            if classification_mode == "incremental"
                            else time_sl[4:]
                        ),
                        classification_mode,
                        bin_dt,
                    )
                    for dataset_result in dataset_results
                ]

                # Split ITR and "shifted" time.
                new_time = [it[1] for it in itrs]
                itrs = [it[0] for it in itrs]

                mx = [np.log(np.max(itr)) for itr in itrs]
                mxi = [np.argmax(itr) for itr in itrs]
                mxt = [np.log(new_time[0][mxj]) for mxj in mxi]

                sub_results.append(mx)
                sub_times.append(mxt)

        # Aggregate results.
        all_trials.append(np.tile(n_trials, len(res_to_load)))
        all_results.append(np.hstack(sub_results))
        all_times.append(np.hstack(sub_times))

        if j == 0:
            print("\n")
            print("Analyzing dataset {}.".format(title_str))
            print(
                "trials: {} - {} (av. {})".format(
                    np.min(all_trials[d]),
                    np.max(all_trials[d]),
                    np.median(all_trials[d]),
                )
            )
            print("\n")

    # ----- #
    # GLMs
    all_datasets = np.hstack(all_datasets)
    all_subjects = np.hstack(all_subjects)
    all_variables = np.hstack(all_variables)
    all_trials = np.hstack(all_trials)
    all_results = np.hstack(all_results)
    all_times = np.hstack(all_times)

    # Full model as pandas dataframe.
    lmm_data = pd.DataFrame(
        np.array(
            [
                all_datasets,
                all_subjects,
                all_variables,
                all_trials,
                all_results,
            ]
        ).T,
        columns=["Dataset", "Subject", "Feature", "Trials", "Accuracy"],
    )

    lmm_data_t = pd.DataFrame(
        np.array(
            [
                all_datasets,
                all_subjects,
                all_variables,
                all_trials,
                all_times,
            ]
        ).T,
        columns=["Dataset", "Subject", "Feature", "Trials", "Time"],
    )

    savepath = "/home/sotpapad/Codes/"

    if acc_time_index == False:
        lmm_data.to_csv(
            savepath
            + "time_res{}_class_res{}_max_score.csv".format(clm_str, fooof_save_str)
        )
        lmm_data_t.to_csv(
            savepath
            + "time_res{}_class_res{}_max_time.csv".format(clm_str, fooof_save_str)
        )
    elif acc_time_index == True:
        lmm_data.to_csv(
            savepath
            + "time_res{}_class_res{}_max_itr.csv".format(clm_str, fooof_save_str)
        )
        lmm_data_t.to_csv(
            savepath
            + "time_res{}_class_res{}_max_itr_time.csv".format(clm_str, fooof_save_str)
        )
