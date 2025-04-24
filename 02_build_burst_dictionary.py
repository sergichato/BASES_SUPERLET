import numpy as np

from burst_space import BurstSpace
from help_funcs import load_exp_variables


# ----- #
# Dataset selection.
data = "dreyer2023"  # "zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014", "dreyer2023"

remove_fooof = False    # True, False

# Mode.
mode = "local"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

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
elif data == "dreyer2023":
    variables_path = "{}dreyer_2023/variables.json".format(basepath)
    channel_ids = [3, 5]


# ----- #
# Loading of dataset-specific variables.
experimental_vars = load_exp_variables(json_filename=variables_path)

subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
if data == "cho2017":
    # Some subjects are not included in the dataset.
    subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

#channel_ids = experimental_vars["channel_ids"]

# ----- #
# Additional variables.
trials_fraction = 1.0
solver = "pca"
if solver == "pca":
    nc = 0.90
elif solver == "csp":
    nc = 8

comps_to_visualize_waves = [3, 4]


# ----- #
# Burst dictionary creation and application of dimensionality reduction.
# Only look into channels C3 and C4.
# Use 'band="mu"' in order to load mu band bursts, instead of beta bursts.
bspace = BurstSpace(
    experimental_vars,
    subjects,
    trials_fraction=trials_fraction,
    channel_ids=channel_ids,
    band="beta",
    remove_fooof=remove_fooof,
    #threshold_feature=True,
    #feature="cycles",
    #percentile=50
)

# Fit all available trials in order to produce the visualization.
# Parameter 'output' has to be set to "plots".
bspace.fit_transform(solver=solver, n_components=nc, output="plots")
bspace.plot_dict(
    subsets="all",
    comps_to_visualize=comps_to_visualize_waves,
    savefigs=True,
    plot_format="pdf"
)
