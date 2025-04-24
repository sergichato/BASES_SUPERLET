import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from burst_space import BurstSpace
from help_funcs import load_exp_variables


# ----- #
# Hyperparameters.
metric = "rocauc"
if metric == "rocauc":
    metric_str = "score"
elif metric == "accuracy":
    metric_str = metric

band = "beta"  # "beta", "mu"
if band == "beta":
    band_letter = "β"
elif band == "mu":
    band_letter = "μ"

limit_hspace = True
if limit_hspace == True:
    hspace_str = "_sel"
elif limit_hspace == False:
    hspace_str = ""

remove_fooof = True
if remove_fooof == True:
    fooof_save_str = ""
elif remove_fooof == False:
    fooof_save_str = "_nfs"

threshold = False  # 0.65
if threshold != False:
    threshold_str = "_thr"
elif threshold == False:
    threshold_str = ""


# ----- #
# Additional variables.
trials_fraction = 1.0
solver = "pca"
if solver == "pca":
    nc = 0.99
elif solver == "csp":
    nc = 8

n_comps_to_vis = 8


# ----- #
# Datasets.
datas = ["zhou2016", "2014004", "2014001", "munichmi", "weibo2014", "cho2017"]

# Mode.
mode = "cluster"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

titles = [
    "Zhou 2016",
    "BNCI 2014-004",
    "BNCI 2014-001",
    "MunichMI",
    "Weibo 2014",
    "Cho 2017",
]

variables_paths = [
    "{}zhou_2016/variables.json".format(basepath),
    "{}2014_004/variables.json".format(basepath),
    "{}2014_001/variables.json".format(basepath),
    "{}munichmi/variables.json".format(basepath),
    "{}weibo_2014/variables.json".format(basepath),
    "{}cho_2017/variables.json".format(basepath),
]

# Channels of interest.
channel_ids = [[3, 5], [0, 2], [3, 5], [4, 8], [3, 5], [3, 5]]

# PCA components.
pca_comps = []

for data, variables_path, cids in zip(datas, variables_paths, channel_ids):
    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    savepath = experimental_vars["dataset_path"]

    subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
    if data == "cho2017":
        # Some subjects are not included in the dataset.
        subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

    ind_dicts = []

    # ----- #
    # Burst dictionary creation and application of dimensionality reduction.
    # Only look into channels C3 and C4.
    # Use 'band="mu"' in order to load mu band bursts, instead of beta bursts.
    for s, _ in enumerate(subjects):
        bspace = BurstSpace(
            experimental_vars,
            subjects[s],
            trials_fraction=trials_fraction,
            channel_ids=cids,
            band="beta",
            remove_fooof=remove_fooof,
        )

        # Fit all available trials in order to produce the visualization.
        # Parameter 'purpose' has to be set to "plotting".
        bspace.fit_transform(solver=solver, n_components=nc, purpose="plotting")

        ind_dicts.append(bspace.drm.components_)

    pca_comps.append(ind_dicts)


# ----- #
# Correlations of components across datasets.
correlation_distances = []

for data_comps in pca_comps:
    # Correlation matrices within dataset.
    data_correlation_matrices = []

    for s, sub_pca_comps in enumerate(data_comps):
        # Comparison to rest of subjects.
        rest_sub_pca_comps = np.delete(np.copy(data_comps), np.arange(s + 1))

        no_of_remain_comps = len(rest_sub_pca_comps)

        for n in range(no_of_remain_comps):
            # Correlation matrices of single subject to the rest,
            # excluding those already computed.
            absolute_cor = np.zeros((n_comps_to_vis, n_comps_to_vis))

            for i in range(n_comps_to_vis):
                for j in range(n_comps_to_vis):
                    orig_comp = sub_pca_comps[i, :]
                    rest_comp = rest_sub_pca_comps[n][j, :]

                    # Absolute correlation.
                    absolute_cor[i, j] = np.abs(np.corrcoef(orig_comp, rest_comp)[0, 1])

                    if threshold != False and absolute_cor[i, j] < threshold:
                        absolute_cor[i, j] = 0.0

            data_correlation_matrices.append(np.eye(n_comps_to_vis) - absolute_cor)
    correlation_distances.append(
        np.mean(
            np.linalg.norm(
                np.repeat(
                    np.eye(n_comps_to_vis)[:, :, np.newaxis],
                    len(data_correlation_matrices),
                    axis=2,
                )
                - np.stack(data_correlation_matrices, axis=2),
                axis=(0, 1),
            )
            # np.trace(np.eye(n_comps_to_vis)) -
            # np.trace(np.stack(data_correlation_matrices, axis=2), axis1=0, axis2=1)
        )
    )


# ----- #
# Average distance to diagonal matrix per dataset.
for title, cor_dist in zip(titles, correlation_distances):
    print(
        "Average (per-subject) distance from identity matrix for the {} dataset is: {}".format(
            title, np.around(cor_dist, 3)
        )
    )
