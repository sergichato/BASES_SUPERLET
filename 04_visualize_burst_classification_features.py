import numpy as np

from moabb.datasets import (
    Zhou2016,
    BNCI2014004,
    BNCI2014001,
    MunichMI,
    Weibo2014,
    Cho2017,
)

from burst_space import BurstSpace
from burst_features import BurstFeatures
from preprocess import (
    load_sub,
    apply_preprocessing,
    Dreyer2023,
)
from help_funcs import load_exp_variables
from time_res_features import compute_power


# ----- #
# Dataset selection.
data = "zhou2016"  # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014", "dreyer2023"

remove_fooof = False    # True, False

# Mode.
mode = "local"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

if data == "zhou2016":
    dataset = Zhou2016()
    variables_path = "{}zhou_2016/variables.json".format(basepath)
    channel_ids = [3, 5]
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
    band_pass = [0, 120]
elif data == "2014004":
    dataset = BNCI2014004()
    variables_path = "{}2014_004/variables.json".format(basepath)
    channel_ids = [0, 2]
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
    band_pass = [0, 120]
elif data == "2014001":
    dataset = BNCI2014001()
    variables_path = "{}2014_001/variables.json".format(basepath)
    channel_ids = [3, 5]
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
    band_pass = [0, 120]
elif data == "munichmi":
    dataset = MunichMI()
    variables_path = "{}munichmi/variables.json".format(basepath)
    channel_ids = [4, 8]
    rereference = False
    zapit = False
    noise_freq = 24.8
    noise_wins = [1, 0.5]
    band_pass = [0, 120]
elif data == "cho2017":
    dataset = Cho2017()
    variables_path = "{}cho_2017/variables.json".format(basepath)
    channel_ids = [3, 5]
    rereference = False
    zapit = False
    noise_freq = 60.0
    noise_wins = [10, 5]
    band_pass = [0, 120]
elif data == "weibo2014":
    dataset = Weibo2014()
    variables_path = "{}weibo_2014/variables.json".format(basepath)
    channel_ids = [3, 5]
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
    band_pass = [0, 95]
elif data == "dreyer2023":
    dataset = Dreyer2023(basepath=basepath+"dreyer_2023/")
    variables_path = "{}dreyer_2023/variables.json".format(basepath)
    channel_ids = [3, 5]
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
    band_pass = [0, 120]


# ----- #
# Loading of dataset-specific variables.
experimental_vars = load_exp_variables(json_filename=variables_path)

subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
if data == "cho2017":
    # Some subjects are not included in the dataset.
    subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

savepath = experimental_vars["dataset_path"]


# ----- #
# Additional variables.
comps_to_analyze = np.arange(1, 6, 1)  # np.arange(1,11,1) #
comps_groups = 5
tf_method = "superlets"
band = "beta"  # "beta", "mu"

comps_to_vis = [3, 4]
rate_computation = "independent"  # "independent", "joint"


# ----- #
# Burst space model.
trials_fraction = 1.0
solver = "pca"
if solver == "pca":
    nc = 0.9
elif solver == "csp":
    nc = 8

# Select 'band="mu"' in order to load mu band bursts, instead of beta bursts.
bspace = BurstSpace(
    experimental_vars,
    subjects,
    trials_fraction=trials_fraction,
    channel_ids=channel_ids,
    band=band,
    remove_fooof=remove_fooof
)
bspace.fit_transform(solver=solver, n_components=nc)

for subject in subjects:

    # Epochs object and power needed for visualization.
    epochs, labels, meta = load_sub(
        subject,
        dataset,
        experimental_vars["tmin"],
        experimental_vars["tmax"],
        experimental_vars["exp_time_periods"][:2],
        savepath,
        band_pass=band_pass,
    )

    epochs, labels, _, _, _ = apply_preprocessing(
        epochs,
        labels,
        meta,
        channels=experimental_vars["channels"],
        zapit=zapit,
        noise_freq=noise_freq,
        noise_wins=noise_wins,
        return_epochs=True,
    )

    epochs_power_dict = compute_power(
        subject,
        epochs,
        labels,
        experimental_vars,
        tf_method,
        band,
        channel_ids,
        zapit,
        noise_freq,
        noise_wins,
        savepath,
        remove_fooof=False
    )

    # Projection of all bursts in new space.
    subject_dictionary, sub_scores_dists, _ = bspace.transform_sub(
        subject, comps_to_analyze, winsorization=[2, 98]
    )

    # Burst features model.
    bclf = BurstFeatures(experimental_vars)

    # Visualization.
    bclf.plot_features(
        subject,
        subject_dictionary,
        sub_scores_dists,
        labels,
        comps_to_analyze,
        comps_to_vis,
        comps_groups,
        solver,
        epochs,
        epochs_power_dict,
        band=band,
        baseline_correction="independent",
        show_splits=False,
        show_stats=False,
        show_sample=False,
        apply_baseline=True,
        rate_computation=rate_computation,
        plot_difference=True,
        plot_format="pdf",
        savefigs=True,
    )
