import numpy as np

from moabb.datasets import (
    Zhou2016,
    BNCI2014004,
    BNCI2014001,
    MunichMI,
    Weibo2014,
    Cho2017,
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances, Shrinkage

from preprocess import Dreyer2023
from burst_space import BurstSpace
from help_funcs import load_exp_variables
from classification_pipelines import classify_fb_power


# ----- #
# Selection of classification hyperparameters.
data = "zhou2016"           # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014", "dreyer2023"
pipe = "csp"                # "simple", "csp", "riemann"
metric = "rocauc"           # "rocauc", "accuracy"
classification_mode = "sliding"  # "trial", "incremental", "sliding"

n_folds = 5

# Mode.
mode = "local"  # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
    reps = 10
    n_jobs = -1
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/"
    if classification_mode == "trial":
        reps = 100
    elif classification_mode == "incremental" or classification_mode == "sliding":
        reps = 10
    n_jobs = 1

# ----- #
# Dataset selection.
if data == "zhou2016":
    dataset = Zhou2016()
    variables_path = "{}zhou_2016/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
elif data == "2014004":
    dataset = BNCI2014004()
    variables_path = "{}2014_004/variables.json".format(basepath)
    channel_ids = [0, 2]
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
elif data == "2014001":
    dataset = BNCI2014001()
    variables_path = "{}2014_001/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
elif data == "munichmi":
    dataset = MunichMI()
    variables_path = "{}munichmi/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = 24.8
    noise_wins = [1, 0.5]
elif data == "cho2017":
    dataset = Cho2017()
    variables_path = "{}cho_2017/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = 60.0
    noise_wins = [10, 5]
elif data == "weibo2014":
    dataset = Weibo2014()
    variables_path = "{}weibo_2014/variables.json".format(basepath)
    rereference = False
    zapit = False
    noise_freq = None
    noise_wins = None
elif data == "dreyer2023":
    dataset = Dreyer2023(basepath=basepath+"dreyer_2023/")
    variables_path = "{}dreyer_2023/variables.json".format(basepath)
    channel_ids = [3, 5]
    rereference = False
    zapit = False
    noise_freq = 50
    noise_wins = [8.0, 4.5]
    band_pass = [0, 120]


# ----- #
# Loading of dataset-specific variables.
experimental_vars = load_exp_variables(json_filename=variables_path)

savepath = experimental_vars["dataset_path"]

subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
if data == "cho2017":
    # Some subjects are not included in the dataset.
    subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

channels = None  # experimental_vars["channels"]
channel_ids = experimental_vars["channel_ids"]

if pipe == "simple":
    if data == "zhou2016":
        channel_ids = [3, 5]
    elif data == "2014004":
        channel_ids = [0, 2]
    elif data == "2014001":
        channel_ids = [3, 5]
    elif data == "munichmi":
        channel_ids = [4, 8]
    elif data == "cho2017":
        channel_ids = [3, 5]
    elif data == "weibo2014":
        channel_ids = [3, 5]
    elif data == "dryer2023":
        channel_ids = [3, 5]


# ----- #
# Classification initialization.

print(
    "\nDataset: {},\nPipeline: {},\nMetric: {},\nClassification mode: {}.\n".format(
        data, pipe, metric, classification_mode
    )
)

if pipe == "simple":
    pipeline = LDA()
elif pipe == "csp":
    csp = CSP(n_components=4, reg=None, log=True, transform_into="average_power")
    clf = LDA()
    pipeline = [csp, clf]
elif pipe == "riemann":
    # mdm = MDM(metric=dict(mean="riemann", distance="riemann"))
    tgsp = TangentSpace(metric="riemann")
    clf = LDA()
    pipeline = [Covariances(), Shrinkage(), tgsp, clf]


# ----- #
# Estimate filter bank and classification.

filter_banks = [
    [[6, 15]],
    [[15, 30]],
    [[6, 30]],
    [[6, 9], [9, 12], [12, 15]],
    [[15, 18], [18, 21], [21, 24], [24, 27], [27, 30]],
    [[6, 9], [9, 12], [12, 15], [15, 18], [18, 21], [21, 24], [24, 27], [27, 30]],
]

for filter_bank in filter_banks:
    print(
        "Filter bank: {} to {} Hz, in {} bands, with {} Hz step.".format(
            filter_bank[0][0],
            filter_bank[-1][1],
            len(filter_bank),
            filter_bank[0][1] - filter_bank[0][0],
        )
    )

    # Classification.
    classify_fb_power(
        subjects,
        dataset,
        data,
        experimental_vars,
        filter_bank,
        channels,
        channel_ids,
        zapit,
        noise_freq,
        noise_wins,
        savepath,
        pipe,
        pipeline,
        metric=metric,
        reps=reps,
        classification_mode=classification_mode,
        n_folds=n_folds,
        n_jobs=n_jobs,
    )
