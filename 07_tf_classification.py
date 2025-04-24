import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from help_funcs import load_exp_variables
from classification_pipelines import classify_tf_features


# ----- #
# Selection of classification hyperparameters.
data = "zhou2016"       # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014"
metric = "rocauc"       # "rocauc", "accuarcy"
band = "beta"           # "beta", "mu"
remove_fooof = False    # True, False

stratification = 5

# Upper bound for number of groups and number of features.
n_groups = 9
n_chars_tk = 5  # 1: volume, 2: amplitude, 5: rest of tf features

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


# ----- #
# Classification initialization.
classifier = LDA()

print(
    "Dataset: {}, metric: {}, tf-features: {}, removing aperiodic activity: {}".format(
        data, metric, n_chars_tk, remove_fooof
    )
)

if data != "physionet":
    classify_tf_features(
        subjects,
        experimental_vars,
        channel_ids,
        savepath,
        classifier,
        band=band,
        remove_fooof=remove_fooof,
        metric=metric,
        stratification=stratification,
        reps=reps,
        n_chars_tk=n_chars_tk,
        n_groups=n_groups,
        n_folds=5,
        trials_fraction=1.0,
        n_jobs=n_jobs,
    )
else:
    classify_tf_features(
        subjects,
        experimental_vars,
        channel_ids,
        savepath,
        classifier,
        band=band,
        remove_fooof=remove_fooof,
        metric=metric,
        stratification=1,
        reps=reps,
        n_chars_tk=n_chars_tk,
        n_groups=n_groups,
        n_folds=2,
        trials_fraction=1.0,
        n_jobs=n_jobs,
    )
