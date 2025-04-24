import numpy as np

from os.path import join

from burst_space import BurstSpace
from help_funcs import load_exp_variables


# ----- #
# Dataset selection.
datas = ["zhou2016", "2014004", "2014001", "munichmi", "cho2017", "weibo2014", "dreyer2023"]

# Number of repetitions.
reps = 30

# Components to use.
cta = [2, 3, 4, 5, 6, 7, 8]

# Number of groups per component.
n_groups = 7

# Number of components to be retained for the analysis.
n_comps = 3

# Mode.
mode = "local"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

# Compute out-of-sample explained variance?
test_data_variance = True   # True, False
tss = 1.0


# ----- #
# Function that computes sum of explained variance of the
# selected components, with optional support for variance
# of unseen data.
def sum_expl_var(
    model,
    analyzed_components,
    selected_components,
    test_data=None,
):

    # Find selected components amongst those analyzed.
    components = np.array(analyzed_components)[selected_components]

    # Explained variance of model.
    try:
        explained_variance_model = np.sum(model.explained_variance_ratio_[components - 1])
    except:
        # In case the model was not built using PCA, explained
        # variance is not supported out of the box.
        explained_variance_model = 0

    # If no test data is provided it's set to 0.
    if type(test_data) == type(None):
        explained_variance_testset = 0
    
    # Explained variance of selected components on test data.
    elif type(test_data) == np.ndarray:

        explained_variance_testset = []

        # Transform test dataset.
        trans_data = model.transform(test_data)

        for comp in (components - 1):

            # Reconstruct dataset along specific axis.
            comp_data = np.zeros_like(trans_data)
            comp_data[:,comp] = trans_data[:,comp]

            reconstr_data = model.inverse_transform(comp_data)

            # Explained variance of specif compoents as norm
            # difference to test data.
            explained_variance_testset.append(
                1 - (np.linalg.norm(reconstr_data - test_data) ** 2 /
                    np.linalg.norm(test_data - model.mean_) ** 2)
            )
        
        explained_variance_testset = np.sum(explained_variance_testset)
    
    explained_variance = [explained_variance_model, explained_variance_testset]
    
    return explained_variance


# ----- #
# Data parsing.
for data in datas:

    print("\nAnalyzing dataset: {}...".format(data))

    if data == "zhou2016":
        variables_path = "{}zhou_2016/variables.json".format(basepath)
    elif data == "2014004":
        variables_path = "{}2014_004/variables.json".format(basepath)
    elif data == "2014001":
        variables_path = "{}2014_001/variables.json".format(basepath)
    elif data == "munichmi":
        variables_path = "{}munichmi/variables.json".format(basepath)
    elif data == "cho2017":
        variables_path = "{}cho_2017/variables.json".format(basepath)
    elif data == "weibo2014":
        variables_path = "{}weibo_2014/variables.json".format(basepath)
    elif data == "dreyer2023":
        variables_path = "{}dreyer_2023/variables.json".format(basepath)


    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    savepath = experimental_vars["dataset_path"]

    subjects = np.arange(1, experimental_vars["n_subjects"] + 1, 1).tolist()
    if data == "cho2017":
        # Some subjects are not included in the dataset.
        subjects = np.delete(np.array(subjects), [31, 45, 48]).tolist()

    channel_ids = experimental_vars["channel_ids"]


    # ----- #
    # Whole data is loaded as a test set.
    if test_data_variance == True:
        print("\nTest data sample size: 1.0...\n")
        bspace = BurstSpace(
            experimental_vars,
            subjects,
            trials_fraction=tss,
            channel_ids=channel_ids,
            band="beta",
            remove_fooof=False,
        )
        bspace.fit_transform(n_components=cta[-1], output="waveforms")
        all_bursts = bspace.all_subs_bursts


    # ----- #
    # Sample size.
    if data != "dreyer2023":
        trials_fractions = [0.05, 0.10, 0.15]
    else:
        trials_fractions = [0.025, 0.05]


    # ----- #
    for trials_fraction in trials_fractions:

        print("\nSample size: {}".format(trials_fraction))

        all_components = []
        all_waveforms = []
        explained_variances = []

        for r in range(reps):

            print("\n....Repetition {}/{}...".format(r+1, reps))
        
            # Burst dictionary creation and application of dimensionality reduction.
            bspace = BurstSpace(
                experimental_vars,
                subjects,
                trials_fraction=trials_fraction,
                channel_ids=channel_ids,
                band="beta",
                remove_fooof=False,
            )

            # Fit all available trials.
            bspace.fit_transform(n_components=cta[-1], output="waveforms")

            # Compute components and corresponding waveforms.
            drm_components, binned_waveforms, _, selected_comps = bspace.estimate_waveforms(
                cta,
                n_groups,
                n_comps=n_comps,
            )

            all_components.append(drm_components)

            binned_waveforms = np.stack(binned_waveforms).reshape(
                n_comps * len(binned_waveforms[0]),
                binned_waveforms[0][0].shape[0]
            )
            all_waveforms.append(binned_waveforms)

            # Compute explained variance.
            explained_variance = sum_expl_var(
                bspace.drm,
                cta,
                selected_comps,
                test_data=all_bursts if test_data_variance == True else None,
            )
            explained_variances.append(explained_variance)
        
        # Save results per dataset and sample size.
        np.save(
            join(
                savepath,
                "eigenvectors_ss_{}.npy".format(trials_fraction)
            ),
            all_components
        )

        np.save(
            join(
                savepath,
                "waveforms_ss_{}.npy".format(trials_fraction)
            ),
            all_waveforms
        )

        np.save(
            join(
                savepath,
                "explained_variances_ss_{}.npy".format(trials_fraction)
            ),
            np.vstack(explained_variances)
        )