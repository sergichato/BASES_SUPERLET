import numpy as np
from os.path import join

from moabb.datasets import (
    Zhou2016,
    BNCI2014004,
    BNCI2014001,
    MunichMI,
    Weibo2014,
    Cho2017,
)

from burst_analysis import TfBursts
from preprocess import (
    load_sub,
    apply_preprocessing,
    Dreyer2023,
)
from help_funcs import load_exp_variables


# ----- #
# Dataset selection.
data = "dreyer2023"  # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014", "dreyer2023"

# Mode.
mode = "local"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

print("Dataset: {}".format(data))

if data == "zhou2016":
    dataset = Zhou2016()
    variables_path = "{}zhou_2016/variables.json".format(basepath)
    band_pass = [0, 120]
elif data == "2014004":
    dataset = BNCI2014004()
    variables_path = "{}2014_004/variables.json".format(basepath)
    band_pass = [0, 120]
elif data == "2014001":
    dataset = BNCI2014001()
    variables_path = "{}2014_001/variables.json".format(basepath)
    band_pass = [0, 120]
elif data == "munichmi":
    dataset = MunichMI()
    variables_path = "{}munichmi/variables.json".format(basepath)
    zapit = True
    noise_freq = 24.8
    noise_wins = [1, 0.5]
    band_pass = [0, 120]
elif data == "cho2017":
    dataset = Cho2017()
    variables_path = "{}cho_2017/variables.json".format(basepath)
    zapit = True
    noise_freq = 60.0
    band_pass = [0, 120]
elif data == "weibo2014":
    dataset = Weibo2014()
    variables_path = "{}weibo_2014/variables.json".format(basepath)
    band_pass = [0, 95]
elif data == "dreyer2023":
    dataset = Dreyer2023(basepath=basepath+"dreyer_2023/")
    variables_path = "{}dreyer_2023/variables.json".format(basepath)
    zapit = True
    noise_freq = 50
    noise_wins = [8.0, 4.5]
    band_pass = [0, 120]


# ----- #
# Loading of dataset-specific variables.
experimental_vars = load_exp_variables(json_filename=variables_path)

subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
if data == "cho2017":
    # Some subjects are not included in the dataset.
    subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

channels = experimental_vars["channels"]
channel_ids = experimental_vars["channel_ids"]

tmin = experimental_vars["tmin"]
tmax = experimental_vars["tmax"]
exp_time_periods = experimental_vars["exp_time_periods"]

savepath = experimental_vars["dataset_path"]


# ----- #
# Frequency axis and indices for wavelets and/or superlets analysis.
freq_step = 0.5
freqs = np.arange(1.0, 43.25, freq_step)

upto_gamma_band = np.array([1, 40])
upto_gamma_range = np.where(
    np.logical_and(freqs >= upto_gamma_band[0], freqs <= upto_gamma_band[1])
)[0]


# ----- #
# Burst extraction.
# By default it produces a visualization for 2 channels corresponding to C3 and C4.
# Select 'produce_plots=False' to skip this step.
# Select 'remove_fooof=False' to skip the subtraction of the aperiodic fit during
# burst detection.
# Select 'tf_method="lagged_coherence"' to work with lagged coherence measures
# instead of TF as a base for detecting bursts (recommended for the mu band).
bm = TfBursts(
    experimental_vars,
    freqs,
    upto_gamma_band,
    upto_gamma_range,
    tf_method="superlets",
    plot_format="png",
    remove_fooof=False,
)

for subject in subjects:
    
    print("Subject {}".format(subject))
    sub_dir = join(savepath, "sub_{}/".format(subject))

    # 1. Subject's raw data loading.
    print("Loading raw data...")

    epochs, labels, meta = load_sub(
        subject,
        dataset,
        tmin,
        tmax,
        exp_time_periods[:2],
        savepath,
        band_pass=band_pass,
    )

    # 2. Pre-processing.
    print("Applying pre-processing...")
    if data == "munichmi" or data == "dreyer2023":
        epochs, labels, meta, _, ntrials = apply_preprocessing(
            epochs,
            labels,
            meta,
            channels,
            zapit=zapit,
            noise_freq=noise_freq,
            noise_wins=noise_wins,
        )
    elif data == "cho2017":
        epochs, labels, meta, _, ntrials = apply_preprocessing(
            epochs, labels, meta, channels, zapit=zapit, noise_freq=noise_freq
        )
    else:
        epochs, labels, meta, _, ntrials = apply_preprocessing(
            epochs, labels, meta, channels
        )

    # Save basic variables for future reference.
    print("Saving dataset meta-information for future reference...")
    np.save(sub_dir + "ntrials", ntrials)
    np.save(sub_dir + "labels", labels)
    np.save(sub_dir + "meta", meta)

    # 3. Burst extraction pipeline.
    # Select 'band="beta"' (default) in order to detect beta bursts, or
    # 'band="mu"' for mu bursts.
    bm.burst_extraction(subject, epochs, labels, band="beta")
    print("\n")
