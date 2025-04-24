# BeBOP BCI

**Beta Bursts Occurrence Patterns for Brain-Computer Interfaces**

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![license: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blueviolet)](https://www.gnu.org/licenses/gpl-3.0.html)


## Description

BeBOP BCI is a small meta-package that aims to provide an alternative way of
looking at how the beta band activity is modulated during motor or kinesthetic
motor imagery tasks, and, consequently, how this modulation can be exploited in
the context of BCI.

Unlike the standard view that solely focuses on signal power, the hypothesis
behind this project is that in terms of neurophysiology it is the modulation of
beta band burst activity that is reflected in changes of power.

This package streamlines the process of analysing the modulation of beta band
burst activity, and constructing features suitable for binary classification
tasks.

The pipeline comprises three steps:
 - detection of burst activity in brain recordings, based on time-frequency
   transforms of the data,
 - analysis of the burst activity as patterns of burst rate modulation for
 bursts of different shapes,
 -  creation of features suitable for binary classification tasks (based on the
 previous step).

The package is build around:
 - the [superlets algorithm](https://www.nature.com/articles/s41467-020-20539-9)
 for decomposing time domain signals in the time-frequency domain, and its
 [python implementation by Gregor Mönke](https://github.com/tensionhead),
 - the [burst detection algorithm](https://github.com/danclab/burst_detection)
 developed by the [DANC lab](https://www.danclab.com/),
 - [MNE-python](https://mne.tools/dev/index.html),
 - the [MOABB project](http://moabb.neurotechx.com/docs/index.html), for simple
 integration with open datasets.


### Requirements

Please refer to [requirements.txt](./requirements.txt) for a full list of
dependencies.


## References

To cite this work (code and/or results):

 - [S. Papadopoulos, et al., Surfing beta bursts to improve
   motor imagery-based BCI. Imaging Neuroscience 2024; 2 1–15](https://doi.org/10.1162/imag_a_00391).

 - [S. Papadopoulos, et al., Improved motor imagery decoding
   with spatiotemporal filtering based on beta burst kernels.
   Proceedings of the 9th Graz Brain-Computer Interface Conference 2024,
   10.3217/978-3-99161-014-4-042](https://www.tugraz.at/fileadmin/user_upload/Institute/INE/Proceedings/Proceedings_GBCIC2024.pdf)

 - [S. Papadopoulos, et al., Beta bursts question the ruling power for
   brain-computer interfaces. J. Neural Eng., 2024, 21 (1), pp.016010](https://iopscience.iop.org/article/10.1088/1741-2552/ad19ea).

Relevant publications (concept and/or results):
 - [S. Papadopoulos, J. Bonaiuto, J. Mattout, An Impending Paradigm Shift
   in Motor Imagery Based Brain-Computer Interfaces. Front. Neurosci. 15 (2022)](https://www.frontiersin.org/articles/10.3389/fnins.2021.824759/full).

Burst detection algorithm:
 - [M. J. Szul, et al., Diverse beta burst waveform motifs characterize
   movement-related cortical dynamics Prog. Neurobiol. 165187](https://www.sciencedirect.com/science/article/abs/pii/S0301008223000916?CMX_ID=&SIS_ID=&dgcid=STMJ_AUTH_SERV_PUBLISHED).


## License

This package is licensed under the
[GPL-3.0-or-later-license](https://www.gnu.org/licenses/gpl-3.0.html).


## Project Status and Roadmap

This project is in a stable state and is actively developed.

This project **is not** meant to, and **will never be** a replacement for a
complete pipeline for analyzing neurophysiological data; it provides basic
integration with MOABB and a simple pre-processing step for illustration
purposes. **When using it with your data, you are expected to develop your
own pre-processing analysis, and classification pipeline.**

You can take advantage of the function calls that instantiate and analyze beta
band burst activity given that you respect the expected structure of the
filesystem. See the examples for more information.

Given the above, this project should be consider in 'beta' development stage
(pun intended).

Future directions include:
 - ~~the ability to meaningfully combine data from more channels, while avoiding
   overfitting,~~
 - ~~the development of a full-fledged feature selection process,~~
 
and possibly:
 - the inclusion of the alpha/mu frequency band in the analysis pipeline *(at
   which point the development stage will be considered as 'alpha' :P )*,
 - support for burst detection not based on a time-frequency
   decomposition method.


## Usage / Documentation

Please refer to the [project's wiki](https://gitlab.com/sotpapad/bebopbci/-/wikis/home)
for documentation on the package's basic API calls. Follow along the provided
examples in order to understand how to use the package and replicate the
results.


## Contributing

The project is open to contributions (well-documented and with minimal dependencies)! 


## Funding

This project is funded by the following agencies:
 - Agence nationale de la recherche (ANR)
 - European Research Council (ERC)
