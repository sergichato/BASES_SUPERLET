import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from matplotlib.style import use

from mne.time_frequency import tfr_array_morlet
from superlet_mne import superlets_mne_epochs

from moabb.datasets import (
    Zhou2016,
    BNCI2014004,
    BNCI2014001,
    MunichMI,
    Weibo2014,
    Cho2017,
)
from preprocess import (
    load_sub,
    apply_preprocessing,
    Dreyer2023,
)
from help_funcs import load_exp_variables

use("default")


# ----- #
# Dataset selection.
datas = ["cho2017", "munichmi", "dreyer2023"]
zapits = [False, True]

# Mode.
mode = "local"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

savefigs = True  # True, False

plot_format = "pdf"  # "pdf", "png"


# ----- #
# Figure initialization.
screen_res = [1920, 972]
dpi = 300

if savefigs == False:
    fig = plt.figure(
        constrained_layout=True,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
else:
    fig = plt.figure(
        constrained_layout=False,
        figsize=(7, 3 * len(datas)),
        dpi=dpi,
    )

gs = fig.add_gridspec(
    nrows=len(datas),
    ncols=2,
    hspace=0.30,
    wspace=0.35,
    left=0.07,
    right=0.93,
)


# ----- #
# Frequency axis and indices for wavelets and/or superlets analysis.
freq_step = 0.5
freqs = np.arange(1.0, 43.25, freq_step)
n_cycles = freqs / 2.0

upto_gamma_band = np.array([1, 40])
upto_gamma_range = np.where(
    np.logical_and(freqs >= upto_gamma_band[0], freqs <= upto_gamma_band[1])
)[0]

beta_band = [15, 30]
beta_freqs = np.where(
    (freqs >= beta_band[0]) & (freqs <= beta_band[1])
)[0]


# ----- #
for d, data in enumerate(datas):

    # Subject selection.
    if data == "munichmi":
        dataset = MunichMI()
        dataset_name = "MunichMI (Grosse-Wentrup 2009)"
        variables_path = "{}munichmi/variables.json".format(basepath)
        noise_freq = 24.8
        noise_wins = [1, 0.5]
        band_pass = [0, 120]
        subject = 2 # any
    elif data == "cho2017":
        dataset = Cho2017()
        dataset_name = "Cho 2017"
        variables_path = "{}cho_2017/variables.json".format(basepath)
        noise_freq = 60.0
        band_pass = [0, 120]
        subject = 1 # 3, 14, 23, 43
    elif data == "dreyer2023":
        dataset = Dreyer2023(basepath=basepath+"dreyer_2023/")
        dataset_name = "Dreyer 2023"
        variables_path = "{}dreyer_2023/variables.json".format(basepath)
        noise_freq = 50
        noise_wins = [8.0, 4.5]
        band_pass = [0, 120]
        subject = 13 # 25
    

    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    if data != "munichmi":
        channels = experimental_vars["channels"]
    elif data == "munichmi":
        channels = experimental_vars["_channels"]
    channel_ids = experimental_vars["channel_ids"]

    c3 = int(np.where((np.array(channels) == "C3"))[0])
    c4 = int(np.where((np.array(channels) == "C4"))[0])

    sfreq = experimental_vars["sfreq"]

    tmin = experimental_vars["tmin"]
    tmax = experimental_vars["tmax"]
    exp_time_periods = experimental_vars["exp_time_periods"]

    savepath = experimental_vars["dataset_path"]

    print("Dataset: {}, Subject {}".format(dataset_name, subject))
    sub_dir = join(savepath, "sub_{}/".format(subject))
    
    # Time axis.
    exp_time = np.linspace(
        tmin,
        tmax,
        int((np.abs(tmax - tmin)) * sfreq) + 1,
    )
    exp_time = np.around(exp_time, decimals=3)
    
    try:
        baseline_begin = int(np.where(exp_time == exp_time_periods[0])[0])
    except:
        baseline_begin = np.where(exp_time >= exp_time_periods[0])[0][0]
    try:
        task_begin = int(np.where(exp_time == exp_time_periods[1])[0])
    except:
        task_begin = np.where(exp_time >= exp_time_periods[1])[0][0]
    try:
        task_end = int(np.where(exp_time == exp_time_periods[2])[0])
    except:
        task_end = np.where(exp_time >= exp_time_periods[2])[0][0]
    try:
        rebound_end = int(np.where(exp_time == exp_time_periods[3])[0])
    except:
        rebound_end = np.where(exp_time <= exp_time_periods[3])[0][-1]

    erds_time_lims = [baseline_begin, rebound_end]

    erds_time = exp_time[erds_time_lims[0] : erds_time_lims[1] + 1]
    

    # ----- #
    # Subgrids and axes.
    gsd0 = gs[d,0].subgridspec(2, 3, hspace=0.15, wspace=0.10, width_ratios=[0.975, 0.975, 0.05])
    ax000 = fig.add_subplot(gsd0[0, 0])
    ax001 = fig.add_subplot(gsd0[0, 1])
    ax002 = fig.add_subplot(gsd0[0, 2])
    ax010 = fig.add_subplot(gsd0[1, 0])
    ax011 = fig.add_subplot(gsd0[1, 1])
    ax012 = fig.add_subplot(gsd0[1, 2])
    
    gsd1 = gs[d,1].subgridspec(2, 3, hspace=0.15, wspace=0.10, width_ratios=[0.975, 0.975, 0.05])
    ax100 = fig.add_subplot(gsd1[0, 0])
    ax101 = fig.add_subplot(gsd1[0, 1])
    ax102 = fig.add_subplot(gsd1[0, 2])
    ax110 = fig.add_subplot(gsd1[1, 0])
    ax111 = fig.add_subplot(gsd1[1, 1])
    ax112 = fig.add_subplot(gsd1[1, 2])

    # Axes titles.
    ax000.set_title("{}, S{}".format(dataset_name, subject), fontweight="bold", fontsize=6, loc="left")

    # X axes labels.
    for ax in [ax000, ax001, ax010, ax011, ax100, ax101, ax110, ax111]:
        ax.set_xticks(np.arange(tmin, tmax, 1 if data != "munichmi" else 2))
        ax.set_xticklabels([])
    
    # Y axes labels.
    for ax in [ax000, ax010, ax100, ax110]:
        ax.set_ylabel("Frequency (Hz)", fontsize=6)
        ax.set_yticks(np.arange(15, 31, 5))

    for ax in [ax001, ax011, ax101, ax111]:
        ax.set_yticks(np.arange(15, 31, 5))
        ax.set_yticklabels([])

    # Remove unused colorbar axes.
    for ax in [ax002, ax012]:
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])


    # ----- #
    # Pre-processing without and with artifact cleaning.
    noisy_wvlts = []
    clean_wvlts = []
    noisy_sprlts = []
    clean_sprlts = []
    
    for zapit in zapits:

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
            epochs, _, _, _, ntrials = apply_preprocessing(
                epochs,
                labels,
                meta,
                channels,
                zapit=zapit,
                noise_freq=noise_freq,
                noise_wins=noise_wins,
            )
        elif data == "cho2017":
            epochs, _, _, _, ntrials = apply_preprocessing(
                epochs, labels, meta, channels, zapit=zapit, noise_freq=noise_freq
            )
        
        # 3. TFs.        
        # Wavelet transform.
        tfs_wvlts = []
        for t in range(epochs.shape[0]):
            trial = np.copy(epochs[t, :, :])
            trial = trial.reshape(1, trial.shape[0], trial.shape[1])
            tfs_wvlts.append(
                tfr_array_morlet(
                    trial,
                    sfreq=sfreq,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    output="power",
                    n_jobs=-1,
                )
            )
        tfs_wvlts = np.array(np.squeeze(tfs_wvlts))[:,[c3, c4],:,:]
        tfs_wvlts = np.mean(tfs_wvlts[:, :, beta_freqs, :], axis=0)
        
        if zapit == False:
            noisy_wvlts.append(tfs_wvlts)
        else:
            clean_wvlts.append(tfs_wvlts)

        # Superlets transform.
        tfs_sprlts = superlets_mne_epochs(np.copy(epochs[:, [c3,c4], :]), freqs, n_jobs=-1)
        tfs_sprlts = np.mean(tfs_sprlts[:, :, beta_freqs,:], axis=0)

        if zapit == False:
            noisy_sprlts.append(tfs_sprlts)
        else:
            clean_sprlts.append(tfs_sprlts)

        # X axes labels.
        for ax in [ax010, ax011, ax110, ax111]:
            ax.set_xlabel("Time (s)", fontsize=6)
            ax.set_xticks(np.arange(tmin, tmax, 1 if data != "munichmi" else 2))
            ax.set_xticklabels(np.arange(tmin, tmax, 1 if data != "munichmi" else 2))
    

    # ----- #
    # Plots.
    # Wavelets, C3 - C4, before cleaning.
    ax000.imshow(
        noisy_wvlts[0][0, :, erds_time_lims[0] : erds_time_lims[1] + 1],
        aspect="auto",
        origin="lower",
        vmin=np.min(noisy_wvlts),
        vmax=np.max(noisy_wvlts),
        extent=(
            erds_time[0],
            erds_time[-1],
            freqs[beta_freqs[0]],
            freqs[beta_freqs[-1]],
        )
    )
    im001 = ax001.imshow(
        noisy_wvlts[0][1, :, erds_time_lims[0] : erds_time_lims[1] + 1],
        aspect="auto",
        origin="lower",
        vmin=np.min(noisy_wvlts),
        vmax=np.max(noisy_wvlts),
        extent=(
            erds_time[0],
            erds_time[-1],
            freqs[beta_freqs[0]],
            freqs[beta_freqs[-1]],
        )
    )

    # Wavelets, C3 - C4, after cleaning.
    ax100.imshow(
        clean_wvlts[0][0, :, erds_time_lims[0] : erds_time_lims[1] + 1],
        aspect="auto",
        origin="lower",
        vmin=np.min(clean_wvlts),
        vmax=np.max(clean_wvlts),
        extent=(
            erds_time[0],
            erds_time[-1],
            freqs[beta_freqs[0]],
            freqs[beta_freqs[-1]],
        )
    )
    im101 = ax101.imshow(
        clean_wvlts[0][1, :, erds_time_lims[0] : erds_time_lims[1] + 1],
        aspect="auto",
        origin="lower",
        vmin=np.min(clean_wvlts),
        vmax=np.max(clean_wvlts),
        extent=(
            erds_time[0],
            erds_time[-1],
            freqs[beta_freqs[0]],
            freqs[beta_freqs[-1]],
        )
    )

    # Superlets, C3 - C4, before cleaning.
    ax010.imshow(
        noisy_sprlts[0][0, :, erds_time_lims[0] : erds_time_lims[1] + 1],
        aspect="auto",
        origin="lower",
        vmin=np.min(noisy_sprlts),
        vmax=np.max(noisy_sprlts),
        extent=(
            erds_time[0],
            erds_time[-1],
            freqs[beta_freqs[0]],
            freqs[beta_freqs[-1]],
        )
    )
    im011 = ax011.imshow(
        noisy_sprlts[0][1, :, erds_time_lims[0] : erds_time_lims[1] + 1],
        aspect="auto",
        origin="lower",
        vmin=np.min(noisy_sprlts),
        vmax=np.max(noisy_sprlts),
        extent=(
            erds_time[0],
            erds_time[-1],
            freqs[beta_freqs[0]],
            freqs[beta_freqs[-1]],
        )
    )
    
    # Superlets, C3 - C4, after cleaning.
    ax110.imshow(
        clean_sprlts[0][0, :, erds_time_lims[0] : erds_time_lims[1] + 1],
        aspect="auto",
        origin="lower",
        vmin=np.min(clean_sprlts),
        vmax=np.max(clean_sprlts),
        extent=(
            erds_time[0],
            erds_time[-1],
            freqs[beta_freqs[0]],
            freqs[beta_freqs[-1]],
        )
    )
    im111 = ax111.imshow(
        clean_sprlts[0][1, :, erds_time_lims[0] : erds_time_lims[1] + 1],
        aspect="auto",
        origin="lower",
        vmin=np.min(clean_sprlts),
        vmax=np.max(clean_sprlts),
        extent=(
            erds_time[0],
            erds_time[-1],
            freqs[beta_freqs[0]],
            freqs[beta_freqs[-1]],
        )
    )

    # Colorbars.
    plt.colorbar(im001, cax=ax002).set_label(label="Power (a.u.)", size=6)
    plt.colorbar(im011, cax=ax012).set_label(label="Power (a.u.)", size=6)
    plt.colorbar(im101, cax=ax102).set_label(label="Power (a.u.)", size=6)
    plt.colorbar(im111, cax=ax112).set_label(label="Power (a.u.)", size=6)

    for ax in [ax002, ax012, ax102, ax112]:
        ax002.locator_params(nbins=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.yaxis.offsetText.set_fontsize(6)
        

    # Trial onset and tick sizes.
    for ax in [ax000, ax001, ax010, ax011, ax100, ax101, ax110, ax111]:
        ax.axvline(x=exp_time[task_begin], ymin=0, ymax=1, c="w", ls=':')
        ax.axvline(x=exp_time[task_end], ymin=0, ymax=1, c="w", ls=':')
        ax.tick_params(axis="both", labelsize=6)


# ----- #
# Optional saving.
if savefigs == True:
    figname = savepath + "noise_examples.{}".format(plot_format)
    fig.savefig(figname, dpi=dpi, facecolor="w", edgecolor="w")
elif savefigs == False:
    plt.show()