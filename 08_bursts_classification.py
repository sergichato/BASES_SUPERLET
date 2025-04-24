import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from help_funcs import load_exp_variables
from classification_pipelines import classify_waveforms


# ----- #
# Selection of classification hyperparameters.
data = "zhou2016"   # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014"
solver = "pca"      # "pca", "csp"
metric = "rocauc"   # "rocauc", "accuracy"
band = "beta"       # "beta", "mu"
remove_fooof = False  # True, False
limit_hspace = False  # False, True

stratification = 5
keep_significant_features = False  # True, False

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


# Upper bound for number of groups and number of components.
if limit_hspace == False:
    # Explore hyper-parameter space with grid search.
    n_groups = 9
    n_comps_tk = 8

elif limit_hspace == True:
    if remove_fooof == True:
        fooof_save_str = ""
    elif remove_fooof == False:
        fooof_save_str = "_nfs"

    # Use best hyperparameteres per stratum for each subject.
    burst_data = np.load(
        savepath
        + "mean_{}_stratified_{}_bursts_{}{}.npy".format(
            metric, band, solver, fooof_save_str
        )
    )

    n_groups = []
    n_comps_tk = []
    for i in range(burst_data.shape[0]):
        strat_g = []
        strat_nct = []

        for j in range(burst_data.shape[1]):
            ids_g, ids_pc = np.unravel_index(
                np.argmax(burst_data[i, j, :, :]),
                (burst_data.shape[2], burst_data.shape[3]),
            )
            strat_g.append(ids_g)
            strat_nct.append(ids_pc)
        n_groups.append(strat_g)
        n_comps_tk.append(strat_nct)


# ----- #
# Models' variables.
if solver == "pca":
    n_comps = 0.99
elif solver == "csp":
    n_comps = 8


# ----- #
# Classification initialization.
classifier = LDA()

print(
    "Dataset: {}, solver: {}, cross validation: {}, removing aperiodic activity: {}".format(
        data, solver, limit_hspace, remove_fooof
    )
)

if data != "physionet":
    classify_waveforms(
        subjects,
        experimental_vars,
        channel_ids,
        savepath,
        classifier,
        solver=solver,
        band=band,
        remove_fooof=remove_fooof,
        metric=metric,
        reps=reps,
        stratification=stratification,
        n_comps=n_comps,
        n_comps_tk=n_comps_tk,
        n_groups=n_groups,
        keep_significant_features=keep_significant_features,
        limit_hspace=limit_hspace,
        n_jobs=n_jobs,
    )
else:
    classify_waveforms(
        subjects,
        experimental_vars,
        channel_ids,
        savepath,
        classifier,
        solver=solver,
        band=band,
        remove_fooof=remove_fooof,
        metric=metric,
        reps=reps,
        stratification=1,
        n_folds=2,
        n_comps=n_comps,
        n_comps_tk=n_comps_tk,
        n_groups=n_groups,
        keep_significant_features=keep_significant_features,
        limit_hspace=limit_hspace,
        n_jobs=n_jobs,
    )
