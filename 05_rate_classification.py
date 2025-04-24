import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP

from help_funcs import load_exp_variables
from classification_pipelines import classify_feature


# ----- #
# Selection of classification hyperparameters.
data = "zhou2016"           # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014"
pipe = "simple"             # "simple", # "csp"
compute_feature = "rate"    # "rate", "amplitude", "volume", "duration", "fr_span", "peak_fr", "cycles"
band = "beta"               # "band", "mu"
metric = "rocauc"           # "rocauc", "accuracy"
remove_fooof = False        # True, False

classification_mode = "trial"   # "trial", "incremental"

n_folds = 5

# Mode.
mode = "local"  # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
    reps = 10
    n_jobs = -1
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/"
    reps = 100
    n_jobs = 30

# ----- #
# Dataset selection.
if data == "zhou2016":
    variables_path = "{}zhou_2016/variables.json".format(basepath)
    channel_ids = [3, 5]
elif data == "2014004":
    variables_path = "{}2014_004/variables.json".format(basepath)
    channel_ids = [0, 2]
elif data == "2014001":
    variables_path = "{}2014_001/variables.json".format(basepath)
    channel_ids = [3, 5]
elif data == "munichmi":
    variables_path = "{}munichmi/variables.json".format(basepath)
    channel_ids = [4, 8]
elif data == "cho2017":
    variables_path = "{}cho_2017/variables.json".format(basepath)
    channel_ids = [3, 5]
elif data == "weibo2014":
    variables_path = "{}weibo_2014/variables.json".format(basepath)
    channel_ids = [3, 5]


# ----- #
# Loading of dataset-specific variables.
experimental_vars = load_exp_variables(json_filename=variables_path)

savepath = experimental_vars["dataset_path"]

subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
if data == "cho2017":
    # Some subjects are not included in the dataset.
    subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

channels = experimental_vars["channels"]
channel_ids = experimental_vars["channel_ids"]


# ----- #
# Classification initialization.

print(
    "Dataset: {}, pipeline: {}, band: {}, metric: {}, removing aperiodic activity: {}".format(
        data, pipe, band, metric, remove_fooof
    )
)

if pipe == "simple":
    pipeline = LDA()
elif pipe == "csp":
    csp = CSP(n_components=6, reg=None, log=True, transform_into="average_power")
    clf = LDA()
    pipeline = make_pipeline(csp, clf)

classify_feature(
    subjects,
    experimental_vars,
    channel_ids,
    savepath,
    pipe,
    pipeline,
    compute_feature=compute_feature,
    band=band,
    remove_fooof=remove_fooof,
    threshold_feature="amplitude",
    metric=metric,
    reps=reps,
    classification_mode=classification_mode,
    n_folds=n_folds,
    n_jobs=n_jobs,
)
