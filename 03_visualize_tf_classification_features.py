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
from burst_features import TfFeatures
from preprocess import (
    load_sub,
    apply_preprocessing,
    Dreyer2023,
)
from help_funcs import load_exp_variables


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
chars_groups = 4
band = "beta"  # "beta", "mu"


# ----- #
# Burst space model.
trials_fraction = 1.0

# Use 'band="mu"' in order to load mu band bursts, instead of beta bursts.
bspace = BurstSpace(
    experimental_vars,
    subjects,
    trials_fraction=trials_fraction,
    channel_ids=channel_ids,
    band=band,
    remove_fooof=remove_fooof
)

for subject in subjects:

    # Epochs object needed for visualization.
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

    subject_dictionary, sub_chars_dists = bspace.compute_chars_dists(
        subject, winsorization=[2, 98]
    )

    # Burst features model.
    bclf = TfFeatures(experimental_vars)

    # Visualization.
    # Use 'band="mu"' in order to load mu band bursts, instead of beta bursts.
    bclf.plot_features(
        subject,
        dataset,
        subject_dictionary,
        sub_chars_dists,
        labels,
        epochs,
        chars_groups,
        band_pass,
        band=band,
        show_splits=False,
        show_stats=False,
        savefigs=True,
        plot_format="pdf",
    )
