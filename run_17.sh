#!/bin/bash

# ----- #
# Job name
#SBATCH --job-name=Kernel_convolution_decoding


# ----- #
# Jobs to run; each element corresponds to a subject.
# Only run n at a time with (%n) at the end of the command!
# Counting the python way!
# Now using dataset: CHO2017.
#SBATCH --array=0-48%7


# ----- #
# Computational resources.
#SBATCH --cpus-per-task=20
#SBATCH --mem=25G

# Instead of specifying a nodelist which will ask for all the nodes to be
# available for each job, exclude the nodes that are not contained in the
# nodelist. Each job will only occupy one node this way.
# SBATCH --nodelist=node[13-21]
#SBATCH --exclude=node[2-12]


# ----- #
# Task time limit (D-HH:MM:SS)
# SBATCH --time=6:00:00


# ----- #
# Output and error filenames.
# Currently skipped and instead used directly when calling the python script.
# --output=fichier_de_sortie${SLURM_ARRAY_TASK_ID}.txt.txt
# --error=sortie_erreur.err


# ----- #
# Python activation.
# module add Programming_Languages/python/3.9.1

# Activation of virtual python environment.
source /home/sotirios.papadopoulos/virtenv/bin/activate


# ----- #
# Subject name variable.
SUB_ID=$(($SLURM_ARRAY_TASK_ID))
SUB=$(($SUB_ID + 1))


# ----- #
# Run script.
# Standard output and standard error are NOT redirected to the same file.
python3 -u \
/home/sotirios.papadopoulos/bebop_bci/17_burst_convolution_classification.py \
> /mnt/data/sotiris.papadopoulos/output_$SUB.txt \
2> /mnt/data/sotiris.papadopoulos/error_$SUB.txt \
${SUB_ID}
