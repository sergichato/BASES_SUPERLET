import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from matplotlib.style import use

from help_funcs import load_exp_variables

use("default")


# ----- #
# Dataset selection.
datas = ["2014001", "2014004", "cho2017", "dreyer2023", "munichmi", "weibo2014", "zhou2016"]

# Mode.
mode = "local"    # "local", "cluster"
if mode == "local":
    basepath = "/home/sotpapad/Codes/"
elif mode == "cluster":
    basepath = "/mnt/data/sotiris.papadopoulos/" # "/crnldata/cophy/Jeremie/Sotiris/bebop/"

apply_flip = True   # True, False

savefigs = True  # True, False

plot_format = "pdf"  # "pdf", "png"


# ----- #
# Figures initialization.
screen_res = [1920, 972]
dpi = 300

if savefigs == False:
    fig0 = plt.figure(
        constrained_layout=True,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
    fig1 = plt.figure(
        constrained_layout=True,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
    fig2 = plt.figure(
        constrained_layout=True,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
    fig3 = plt.figure(
        constrained_layout=True,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
else:
    fig0 = plt.figure(
        constrained_layout=False,
        figsize=(7, 3 * len(datas)),
        dpi=dpi,
    )
    fig1 = plt.figure(
        constrained_layout=False,
        figsize=(7, 1.2 * len(datas)),
        dpi=dpi,
    )
    fig2 = plt.figure(
        constrained_layout=False,
        figsize=(7, 8),
        dpi=dpi,
    )
    fig3 = plt.figure(
        constrained_layout=False,
        figsize=(7, 9),
        dpi=dpi,
    )

gs0 = fig0.add_gridspec(
    nrows=len(datas),
    ncols=1,
    hspace=0.20,
    left=0.08,
    right=0.95,
    top=0.90,
    bottom=0.10,
)
gs1 = fig0.add_gridspec(
    nrows=len(datas),
    ncols=1,
    hspace=0.50,
    left=0.08,
    right=0.95,
    top=0.90,
    bottom=0.10,
)
gs2 = fig0.add_gridspec(
    nrows=2,
    ncols=1,
    hspace=0.20,
    left=0.08,
    right=0.95,
    top=0.90,
    bottom=0.10,
    height_ratios=[0.05, 0.95]
)
gs3 = fig3.add_gridspec(
    nrows=len(datas)//2 + len(datas)%2 + 2,
    ncols=4,
    hspace=0.50,
    wspace=0.50,
    left=0.10,
    right=0.95,
    top=0.95,
    bottom=0.05,
)

width = 0.5
gap = 0.1

colorlist = ["tab:blue", "tab:orange", "tab:green",
            "tab:purple", "tab:brown", "tab:cyan",
            "tab:pink", "tab:olive", "tab:gray",]

dcolorlist = ["limegreen", "goldenrod", "darkorange", 
                "crimson", "deeppink", "dodgerblue",
                "mediumturquoise",]


marks = ["s", "o", "D", "v", "<", ">", "^",]
ms = 10


# ----- #
# Data parsing.
for d, data in enumerate(datas):


    # ----- #
    print("\nAnalyzing dataset: {}...".format(data))

    if data == "zhou2016":
        variables_path = "{}zhou_2016/variables.json".format(basepath)
        dataset_name = "Zhou 2016"
        sns = 18 * (10**4)
    elif data == "2014004":
        variables_path = "{}2014_004/variables.json".format(basepath)
        dataset_name = "BNCI 2014-004"
        sns = 26 * (10**4)
    elif data == "2014001":
        variables_path = "{}2014_001/variables.json".format(basepath)
        dataset_name = "BNCI 2014-001"
        sns = 37 * (10**4)
    elif data == "munichmi":
        variables_path = "{}munichmi/variables.json".format(basepath)
        dataset_name = "Munich MI (Grosse-Wentrup 2009)"
        sns = 152 * (10**4)
    elif data == "cho2017":
        variables_path = "{}cho_2017/variables.json".format(basepath)
        dataset_name = "Cho 2017"
        sns = 32.5 * (10**4)
    elif data == "weibo2014":
        variables_path = "{}weibo_2014/variables.json".format(basepath)
        dataset_name = "Weibo 2014"
        sns = 11.5 * (10**4)
    elif data == "dreyer2023":
        variables_path = "{}dreyer_2023/variables.json".format(basepath)
        dataset_name = "Dreyer 2023"
        sns = 180 * (10**4)
    

    # ----- #
    # Loading of dataset-specific variables.
    experimental_vars = load_exp_variables(json_filename=variables_path)

    savepath = experimental_vars["dataset_path"]


    # ----- #
    # Sample sizes.
    if data != "dreyer2023":
        trials_fractions = [0.05, 0.10, 0.15]
    else:
        trials_fractions = [0.025, 0.05]
    

    # ----- #
    # Subgrids and axes.
    gsd = gs0[d].subgridspec(1, 2, wspace=0.25, width_ratios=[1.0, 2.0])
    axd0 = fig0.add_subplot(gsd[0])
    axd1 = fig0.add_subplot(gsd[1])

    if d == 0:
        gsd2 = gs2[1].subgridspec(3, 3, wspace=0.25, hspace=0.25)
        axd200 = fig2.add_subplot(gsd2[0,0])
        axd201 = fig2.add_subplot(gsd2[0,1])
        axd202 = fig2.add_subplot(gsd2[0,2])
        axd210 = fig2.add_subplot(gsd2[1,0])
        axd211 = fig2.add_subplot(gsd2[1,1])
        axd212 = fig2.add_subplot(gsd2[1,2])
        axd220 = fig2.add_subplot(gsd2[2,0])
        axd221 = fig2.add_subplot(gsd2[2,1])
        axd222 = fig2.add_subplot(gsd2[2,2])

        axesd20c = [axd200, axd201, axd202]
        axesd21w = [axd210, axd220,
                    axd211, axd221,
                    axd212, axd222]

        gsd33 = gs3[len(datas)//2 + len(datas)%2, :].subgridspec(1, 2, wspace=0.25)
        axd330 = fig3.add_subplot(gsd33[0])
        axd331 = fig3.add_subplot(gsd33[1])

        gsd34 = gs3[len(datas)//2 + len(datas)%2 + 1, :].subgridspec(1, 2, wspace=0.25)
        axd340 = fig3.add_subplot(gsd34[0])
        axd341 = fig3.add_subplot(gsd34[1])

    # Initialize lists to keep track of X axes tick positions.
    c_ticks = []
    w_ticks = []
    

    # ----- #
    # Retrieve data for each sample size.
    comps_coefs_av_all = []
    waves_coefs_av_all = []
    explained_variances_all = []

    for t, trials_fraction in enumerate(trials_fractions):

        print("....Sample size: {}".format(trials_fraction))

        all_components_orig = np.load(
            join(
                savepath,
                "eigenvectors_ss_{}.npy".format(trials_fraction)
            ),
            allow_pickle=True
        )

        all_waveforms_orig = np.load(
            join(
                savepath,
                "waveforms_ss_{}.npy".format(trials_fraction)
            ),
            allow_pickle=True
        )

        explained_variances = np.load(
            join(
                savepath,
                "explained_variances_ss_{}.npy".format(trials_fraction),
            )
        )

        all_components = np.array(all_components_orig)
        comps_shapes = [all_components.shape[0], all_components.shape[1]]
        all_components = all_components.reshape(
            all_components.shape[0] * all_components.shape[1],
            all_components.shape[2]
        )

        all_waveforms = np.array(all_waveforms_orig)
        waves_shapes = [all_waveforms.shape[0], all_waveforms.shape[1]]
        all_waveforms = all_waveforms.reshape(
            all_waveforms.shape[0] * all_waveforms.shape[1],
            all_waveforms.shape[2]
        )

        explained_variances_all.append(explained_variances)

        
        # ----- #
        # Correlations within same sample size.
        comps_coefs = np.corrcoef(all_components)
        waves_coefs = np.corrcoef(all_waveforms)

        comps_coefs_abs = np.abs(comps_coefs)
        waves_coefs_abs = np.abs(waves_coefs)

        # Ranked correlations, averaged across repetitions.
        comps_coefs_av = np.zeros((comps_shapes[1], comps_shapes[1], comps_shapes[0]))
        waves_coefs_av = np.zeros((waves_shapes[1], waves_shapes[1], waves_shapes[0]))

        c_windows = [[comps_shapes[1] * i, comps_shapes[1] * i + comps_shapes[1]] for i in range(comps_shapes[0])]
        w_windows = [[waves_shapes[1] * i, waves_shapes[1] * i + waves_shapes[1]] for i in range(waves_shapes[0])]
        
        for r, row in enumerate(comps_coefs_abs):
            comps_coefs_abs_ranked = np.hstack(
                [-np.sort(-row[window[0]:window[1]]) for window in c_windows]
            )
            comps_coefs_av[r//comps_shapes[0], 0, r%comps_shapes[0]] = np.mean(comps_coefs_abs_ranked[np.copy(c_windows)[:,0]])
            comps_coefs_av[r//comps_shapes[0], 1, r%comps_shapes[0]] = np.mean(comps_coefs_abs_ranked[np.copy(c_windows)[:,0]+1])
            comps_coefs_av[r//comps_shapes[0], 2, r%comps_shapes[0]] = np.mean(comps_coefs_abs_ranked[np.copy(c_windows)[:,0]+2])


        comps_coefs_av = np.mean(comps_coefs_av, axis=-1)

        for r, row in enumerate(waves_coefs_abs):
            waves_coefs_abs_ranked = np.hstack(
                [-np.sort(-row[window[0]:window[1]]) for window in w_windows]
            )
            waves_coefs_av[r//waves_shapes[0], 0, r%waves_shapes[0]] = np.mean(waves_coefs_abs_ranked[np.copy(w_windows)[:,0]])
            waves_coefs_av[r//waves_shapes[0], 1, r%waves_shapes[0]] = np.mean(waves_coefs_abs_ranked[np.copy(w_windows)[:,0]+1])
            waves_coefs_av[r//waves_shapes[0], 2, r%waves_shapes[0]] = np.mean(waves_coefs_abs_ranked[np.copy(w_windows)[:,0]+2])            

        waves_coefs_av = np.mean(waves_coefs_av, axis=-1)
    
        comps_coefs_av_all.append(comps_coefs_av)
        waves_coefs_av_all.append(waves_coefs_av)


        # ----- #
        # Ranked correlations arranged according to first resampling.
        comps_coefs_argmax = np.zeros((comps_shapes[1], comps_shapes[0], 3))
        waves_coefs_argmax = np.zeros((waves_shapes[1], waves_shapes[0], 3))

        eigenvectors = np.zeros((comps_shapes[1], comps_shapes[0], all_components.shape[1]))
        waveforms = np.zeros((waves_shapes[1], waves_shapes[0], all_waveforms.shape[1]))
        
        for r, row in enumerate(comps_coefs[0:comps_shapes[1], :]):

            # Directly sort and store shapes.
            for k, window in enumerate(c_windows):

                comps_coefs_argmax[r,k,0] = np.max(np.abs(row[window[0]:window[1]]))
                comps_coefs_argmax[r,k,1] = np.argmax(np.abs(row[window[0]:window[1]]))
                comps_coefs_argmax[r,k,2] = np.sign(
                    row[window[0]:window[1]][int(comps_coefs_argmax[r,k,1])]
                )
                eigenvectors[r,k,:] = all_components[window[0]:window[1],:][int(comps_coefs_argmax[r,k,1]),:]
            
            rank = np.argsort(
                np.hstack(
                    [-np.max(np.abs(row[window[0]:window[1]])) for window in c_windows]
                )
            )
            comps_coefs_argmax[r,:,0] = comps_coefs_argmax[r,:,0][rank]
            comps_coefs_argmax[r,:,1] = comps_coefs_argmax[r,:,1][rank]
            comps_coefs_argmax[r,:,2] = comps_coefs_argmax[r,:,2][rank]
            eigenvectors[r,:,:] = eigenvectors[r,:,:][rank,:]
        
        for r, row in enumerate(waves_coefs[0:waves_shapes[1], :]):

            for k, window in enumerate(w_windows):

                waves_coefs_argmax[r,k,0] = np.max(np.abs(row[window[0]:window[1]]))
                waves_coefs_argmax[r,k,1] = np.argmax(np.abs(row[window[0]:window[1]]))
                waves_coefs_argmax[r,k,2] = np.sign(
                    row[window[0]:window[1]][int(waves_coefs_argmax[r,k,1])]
                )
                waveforms[r,k,:] = all_waveforms[window[0]:window[1],:][int(waves_coefs_argmax[r,k,1]),:]
            
            rank = np.argsort(
                np.hstack(
                    [-np.max(np.abs(row[window[0]:window[1]])) for window in w_windows]
                )
            )
            waves_coefs_argmax[r,:,0] = waves_coefs_argmax[r,:,0][rank]
            waves_coefs_argmax[r,:,1] = waves_coefs_argmax[r,:,1][rank]
            waves_coefs_argmax[r,:,2] = waves_coefs_argmax[r,:,2][rank]
            waveforms[r,:,:] = waveforms[r,:,:][rank,:]


        # ----- #
        # Plots.

        # X axes positions.
        if t == 0:
            if comps_shapes[1] % 2 == 1:
                c_offset = np.arange(
                    -(width + gap) * (comps_shapes[1] // 2),
                    (width + gap + 0.01) * (comps_shapes[1] // 2),
                    (width + gap),
                )
            elif comps_shapes[1] % 2 == 0:
                c_offset = np.arange(
                    -((width + gap) / 2) * (comps_shapes[1] // 2 + (comps_shapes[1] // 2 - 1)),
                    ((width + gap) / 2 + 0.01) * (comps_shapes[1] // 2 + (comps_shapes[1] // 2 - 1)),
                    (width + gap),
                )
            
            if waves_shapes[1] % 2 == 1:
                w_offset = np.arange(
                    -(width + gap) * (waves_shapes[1] // 2),
                    (width + gap + 0.01) * (waves_shapes[1] // 2),
                    (width + gap),
                )
            elif waves_shapes[1] % 2 == 0:
                w_offset = np.arange(
                    -((width + gap) / 2) * (waves_shapes[1] // 2 + (waves_shapes[1] // 2 - 1)),
                    ((width + gap) / 2 + 0.01) * (waves_shapes[1] // 2 + (waves_shapes[1] // 2 - 1)),
                    (width + gap),
                )
        else:
            c_offset += c_offset[-1] - c_offset[0] + 2*width + gap
            w_offset += w_offset[-1] - w_offset[0] + 2*width + gap
        
        # Keep track of x tick positions.
        c_ticks.append(np.copy(c_offset))
        w_ticks.append(np.copy(w_offset))
        
        # Stack ranks.
        c_bottom = np.zeros(comps_coefs_av.shape[1])
        w_bottom = np.zeros(waves_coefs_av.shape[1])
        
        # Barplots.
        for c_rank in range(comps_coefs_av.shape[1]):

            b0 = axd0.bar(
                c_offset,
                comps_coefs_av[:, c_rank],
                bottom=c_bottom,
                width=width,
                label=str(c_rank + 1) if (t+1) == len(trials_fractions) and d == 0 else "",
                color=colorlist[c_rank],
                zorder=10,
            )
            
            axd0.bar_label(
                b0,
                label_type='center',
                fmt="%0.2f",
                zorder=20,
                fontsize=3,
            )

            # Update rank stack.
            c_bottom += comps_coefs_av[:, c_rank]

        for w_rank in range(waves_coefs_av.shape[1]):

            b1 = axd1.bar(
                w_offset,
                waves_coefs_av[:, w_rank],
                bottom=w_bottom,
                width=width,
                label=str(w_rank + 1) if (t+1) == len(trials_fractions) and d == 0 and w_rank > 2 else "",
                color=colorlist[w_rank],
                zorder=10,
            )
            
            axd1.bar_label(
                b1,
                label_type='center',
                fmt="%0.2f",
                zorder=20,
                fontsize=3,
            )
            
            # Update rank stack.
            w_bottom += waves_coefs_av[:, w_rank]
        
        # Relationship of sample size and Pearson correlation.
        for c in range(comps_shapes[1]):
            axd20 = axesd20c[c]
            axd20.scatter(
                sns * (t + 1),
                comps_coefs_av[c,0],
                s=ms,
                marker=marks[t+1] if data != "dreyer2023" else marks[t],
                edgecolors=dcolorlist[d],
                facecolors="none",
                label=dataset_name if t == 0 and c == 0 else "",
                zorder=10,
            )

        for w in range(waves_shapes[1]):
            axd21 = axesd21w[w]
            axd21.scatter(
                sns * (t + 1),
                waves_coefs_av[w,0],
                s=ms,
                marker=marks[t+1] if data != "dreyer2023" else marks[t],
                edgecolors=dcolorlist[d],
                facecolors="none",
                zorder=10,
            )
        
        # Pearson correlation averaged across eigenvectors and waveforms.
        axd330.scatter(
            sns * (t + 1),
            #np.mean(comps_coefs_av[:,0], axis=-1),
            np.mean(np.sum(comps_coefs_av, axis=-1)),
            s=ms,
            marker=marks[t+1] if data != "dreyer2023" else marks[t],
            edgecolors=dcolorlist[d],
            facecolors="none",
            zorder=10,
        )

        axd331.scatter(
            sns * (t + 1),
            np.mean(waves_coefs_av[:,0], axis=-1),
            #np.mean(np.sum(waves_coefs_av, axis=-1)),
            s=ms,
            marker=marks[t+1] if data != "dreyer2023" else marks[t],
            label=dataset_name if t == 0 else "",
            edgecolors=dcolorlist[d],
            facecolors="none",
            zorder=10,
        )


        # Within- and out-of-sample exaplained variance of selected components.
        axd340.scatter(
            sns * (t + 1),
            np.mean(explained_variances[:,0]),
            s=ms,
            marker=marks[t+1] if data != "dreyer2023" else marks[t],
            edgecolors=dcolorlist[d],
            facecolors="none",
            zorder=10,
        )

        axd341.scatter(
            sns * (t + 1),
            np.mean(explained_variances[:,1]),
            s=ms,
            marker=marks[t+1] if data != "dreyer2023" else marks[t],
            label=dataset_name if t == 0 else "",
            edgecolors=dcolorlist[d],
            facecolors="none",
            zorder=10,
        )
        
        # Selected eigenvectors' and waveforms' shapes.
        if t == 0:
            gsd1 = gs1[d].subgridspec(comps_shapes[1] + waves_shapes[1] + 2, 3 * comps_shapes[0] + 2)
            gsd30 = gs3[d//2, 2 * (d%2)].subgridspec(comps_shapes[1], 3)
            gsd31 = gs3[d//2, 2 * (d%2) + 1].subgridspec(waves_shapes[1], 3)
        
        for c in range(comps_shapes[1]):
            for s in range(comps_shapes[0]):
                axd10t = fig1.add_subplot(gsd1[c, s + (comps_shapes[0] * t + t)])
                axd10t.plot(
                    all_components_orig[s,c,:],
                    linewidth=0.5,
                    c=colorlist[c],
                )
                axd10t.set_axis_off()

                if c == 0 and s == 0 and t == 0:
                    axd10t.set_title("{}".format(dataset_name), fontweight="bold", fontsize=6, loc="left")
                
            axd30 = fig3.add_subplot(gsd30[c, t])
            for eigenvector_shape, comp_coefs_argmax in zip(eigenvectors[c,:,:], comps_coefs_argmax[c,:,:]):
                axd30.plot(
                    eigenvector_shape if apply_flip == False else eigenvector_shape * comp_coefs_argmax[2],
                    linewidth=0.5,
                    c=colorlist[c],
                    alpha=np.around(np.abs(comp_coefs_argmax[0]), decimals=2),
                )

            # Title, labels and spines.
            if c == 0 and t == 0:
                axd30.set_title("{}".format(dataset_name), fontweight="bold", fontsize=6, loc="left")
            if c == comps_shapes[1] - 1:
                axd30.set_xlabel("s: {}%".format(trials_fraction * 100), fontsize=6)
            if t == 0:
                axd30.set_ylabel("e{}".format(c + 1), fontsize=6, rotation=0)
            
            axd30.spines[:].set_visible(False)
            axd30.set_xticks([])
            axd30.set_yticks([])
        
        for w in range(waves_shapes[1]):
            for s in range(comps_shapes[0]):
                axd11t = fig1.add_subplot(gsd1[w + comps_shapes[1] + 2, s + (comps_shapes[0] * t + t)])
                axd11t.plot(
                    all_waveforms_orig[s,w,:],
                    linewidth=0.5,
                    c=colorlist[w + comps_shapes[1]],
                )
                axd11t.set_axis_off()
            
            axd31 = fig3.add_subplot(gsd31[w, t])
            for waveform_shape, wave_coefs_argmax in zip(waveforms[w,:,:], waves_coefs_argmax[w,:,:]):
                axd31.plot(
                    waveform_shape if apply_flip == False else waveform_shape * wave_coefs_argmax[2],
                    linewidth=0.5,
                    c=colorlist[w + comps_shapes[1]],
                    alpha=np.around(np.abs(wave_coefs_argmax[0]), decimals=2),
                )

            # Labels and spines.
            if w == waves_shapes[1] - 1:
                axd31.set_xlabel("s: {}%".format(trials_fraction * 100), fontsize=6)
            if t == 0:
                axd31.set_ylabel("w{}".format(w + 1), fontsize=6, rotation=0)
            
            axd31.spines[:].set_visible(False)
            axd31.set_xticks([])
            axd31.set_yticks([])


    # Connected sample sizes per dataset.
    for c in range(comps_shapes[1]):
        axd20 = axesd20c[c]
        axd20.plot(
            [sns * (k + 1) for k in range(len(trials_fractions))],
            np.stack(comps_coefs_av_all)[:,c,0],
            alpha=0.5,
            linewidth=1.0,
            c=dcolorlist[d],
        )

    for w in range(waves_shapes[1]):
        axd21 = axesd21w[w]
        axd21.plot(
            [sns * (k + 1) for k in range(len(trials_fractions))],
            np.stack(waves_coefs_av_all)[:,w,0],
            alpha=0.5,
            linewidth=1.0,
            c=dcolorlist[d],
        )
    
    axd330.plot(
        [sns * (k + 1) for k in range(len(trials_fractions))],
        #np.mean(np.stack(comps_coefs_av_all)[:,:,0], axis=1),
        np.mean(np.stack(np.sum(comps_coefs_av_all, axis=-1)), axis=1),
        alpha=0.5,
        linewidth=1.0,
        c=dcolorlist[d],
    )

    axd331.plot(
        [sns * (k + 1) for k in range(len(trials_fractions))],
        np.mean(np.stack(waves_coefs_av_all)[:,:,0], axis=1),
        #np.mean(np.stack(np.sum(waves_coefs_av_all, axis=-1)), axis=1),
        alpha=0.5,
        linewidth=1.0,
        c=dcolorlist[d],
    )

    axd340.plot(
        [sns * (k + 1) for k in range(len(trials_fractions))],
        np.mean(np.stack(explained_variances_all)[:,:,0], axis=1),
        alpha=0.5,
        linewidth=1.0,
        c=dcolorlist[d],
    )

    axd341.plot(
        [sns * (k + 1) for k in range(len(trials_fractions))],
        np.mean(np.stack(explained_variances_all)[:,:,1], axis=1),
        alpha=0.5,
        linewidth=1.0,
        c=dcolorlist[d],
    )


    # ----- #
    # Titles, ticks and labels.

    # Fig.
    axd0.set_title("{}".format(dataset_name), fontweight="bold", fontsize=6, loc="left")
    
    for ax in [axd0, axd1]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=6)
        ax.yaxis.grid(True, zorder=1)
    
    if d == len(datas) - 1:
        axd0.set_xlabel("Eigenvector", fontsize=6)
    axd0.set_xticks(np.hstack(c_ticks))
    axd0.set_xticklabels(
        np.tile(
            np.arange(1, comps_shapes[1] + 1, 1),
            len(trials_fractions)
        ),
    )
    
    if d == len(datas) - 1:
        axd1.set_xlabel("Waveform", fontsize=6)
    axd1.set_xticks(np.hstack(w_ticks))
    axd1.set_xticklabels(
        np.tile(
            np.arange(1, waves_shapes[1] + 1, 1),
            len(trials_fractions)
        ),
    )

    axd0.set_ylabel("|Pearson's r|", fontsize=6)
    axd0.set_yticks(np.around(np.arange(0, 0.81, 0.1), 2))
    axd0.set_yticklabels(np.around(np.arange(0, 0.81, 0.1), 2))
    
    axd1.set_yticks(np.around(np.arange(0, 1.81, 0.1), 2))
    axd1.set_yticklabels(np.around(np.arange(0, 1.81, 0.1), 2))

    # Fig2 and Fig3.
    if (d == len(datas) - 1) and (t == len(trials_fractions) - 1):

        axd330.set_title("Eigenvectors", fontweight="bold", fontsize=6, loc="left")
        axd331.set_title("Waveforms", fontweight="bold", fontsize=6, loc="left")

        axd340.set_title("Within-sample", fontweight="bold", fontsize=6, loc="left")
        axd341.set_title("Out-of-sample", fontweight="bold", fontsize=6, loc="left")

        axd340.set_ylabel("Explained variance ratio", fontsize=6)
        
        for ax in [axd220, axd221, axd222, axd330, axd331, axd340, axd341]:
            ax.set_xlabel("(# bursts)", fontsize=6)
        
        for ax in [axd200, axd210, axd220, axd330]:
            ax.set_ylabel("|Pearson's r|", fontsize=6)
        
        for ax in axesd20c + axesd21w + [axd330, axd331, axd340, axd341]:
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="both", labelsize=6)

            ax.set_xscale("log")
            """
            ax.set_xticks(
                np.log10(
                    np.arange(5 * (10**5), 5.1 * (10**6), 5 * (10**5))
                )
            )
            ax.set_xticklabels(
                np.log10(
                    np.arange(0.5, 5.1, 0.5)
                ),
                fontsize=6,
            )
            """
            
            ax.set_yticks(np.around(np.arange(0.4, 0.81, 0.1), 2))
            ax.set_yticklabels(np.around(np.arange(0.4, 0.81, 0.1), 2))
            ax.set_ylim([0.4, 0.85])
            ax.yaxis.grid(True, zorder=1)

            """
            if ax == axd341:
                ax.set_yticks(np.around(np.arange(1.0, 1.61, 0.1), 2))
                ax.set_yticklabels(np.around(np.arange(1.0, 1.61, 0.1), 2))
                ax.set_ylim([1.0, 1.65])
                ax.yaxis.grid(True, zorder=1)
            """

            if ax == axd340 or ax == axd341:
                ax.set_yticks(np.around(np.arange(0.10, 0.41, 0.1), 2))
                ax.set_yticklabels(np.around(np.arange(0.10, 0.41, 0.1), 2))
                ax.set_ylim([0.10, 0.45])
                ax.yaxis.grid(True, zorder=1)


# ----- #
# Figures' legends.
fig0.legend(
    frameon=False,
    title="Rank",
    alignment="left",
    fontsize=6,
    title_fontsize=6,
    ncols=2,
    loc="upper right",
)

axd331.legend(
    frameon=False,
    title="Dataset",
    alignment="left",
    fontsize=6,
    title_fontsize=6,
    ncols=2,
    loc="upper right",
)


# ----- #
# Optional saving.
if savefigs == True:
    figname0 = savepath + "sample_size_impact.{}".format(plot_format)
    fig0.savefig(figname0, dpi=dpi, facecolor="w", edgecolor="w")
    figname1 = savepath + "eigenvectos_and_waveforms.{}".format(plot_format)
    fig1.savefig(figname1, dpi=dpi, facecolor="w", edgecolor="w")
    figname2 = savepath + "sample_size_impact_compact.{}".format(plot_format)
    fig2.savefig(figname2, dpi=dpi, facecolor="w", edgecolor="w")
    figname3 = savepath + "eigenvectos_and_waveforms_compact.{}".format(plot_format)
    fig3.savefig(figname3, dpi=dpi, facecolor="w", edgecolor="w")
elif savefigs == False:
    plt.show()