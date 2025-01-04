#!/usr/bin/env bash
#
# IMPORTANT : PUT THE FILES rh.pial.T1, lh.pial.T1, rh.white, lh.white 
# in the same folder as this script
# TO RUN ON WSL : dos2unix process_reconall.sh
# bash process_reconall.sh
# This script performs the following steps:
#   1) Convert FreeSurfer .white/.pial to GIFTI
#   2) Create mid-thickness surfaces with wb_command
#   3) Resample to ~32k vertices with mris_remesh
#   4) Inflate the mid-thickness surfaces

set -e  # Exit if any command fails

############################
# 0) Set any paths or vars #
############################

# Example: if you have a custom location for Workbench commands, FreeSurfer, etc.
# export PATH=/path/to/workbench/bin_linux64:$PATH
# export FREESURFER_HOME=/path/to/freesurfer
# source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Or do nothing if everything is already in your PATH.


#####################################
# 1) Convert FreeSurfer surfaces    #
#    to GIFTI format                #
#####################################
for subj in {01..18}; do
  # 1) Change directory to each subject's 'surf' folder
    cd /mnt/d/DATA_min_preproc/dataset_study1/sub-${subj}/sub${subj}_freesurfer/surf

    echo "Start processing subject ${subj}..."
    echo "Converting to GIFTI (.gii)..."

    mris_convert lh.white      lh.white.surf.gii
    mris_convert lh.pial.T1    lh.pial.surf.gii
    mris_convert rh.white      rh.white.surf.gii
    mris_convert rh.pial.T1    rh.pial.surf.gii

    echo "Done converting to GIFTI."


    #####################################
    # 2) Create mid-thickness surfaces  #
    #    using wb_command -surface-average
    #####################################

    echo "Creating mid-thickness surfaces..."

    wb_command -surface-average lh.midthickness.surf.gii \
    -surf lh.white.surf.gii \
    -surf lh.pial.surf.gii

    wb_command -surface-average rh.midthickness.surf.gii \
    -surf rh.white.surf.gii \
    -surf rh.pial.surf.gii

    echo "Done creating mid-thickness surfaces."


    #####################################
    # 3) Resample to 32k vertices       #
    #    using mris_remesh             #
    #####################################

    echo "Resampling surfaces to ~32k vertices..."

    mris_remesh -i lh.midthickness.surf.gii -o lh.midthickness.32k.surf.gii --nvert 32492
    mris_remesh -i rh.midthickness.surf.gii -o rh.midthickness.32k.surf.gii --nvert 32492

    mris_remesh -i lh.pial.surf.gii -o lh.pial.32k.surf.gii --nvert 32492
    mris_remesh -i rh.pial.surf.gii -o rh.pial.32k.surf.gii --nvert 32492

    mris_remesh -i lh.white.surf.gii -o lh.white.32k.surf.gii --nvert 32492
    mris_remesh -i rh.white.surf.gii -o rh.white.32k.surf.gii --nvert 32492

    echo "Done resampling surfaces to 32k."


    #####################################
    # 4) Inflate the mid-thickness      #
    #    surfaces with mris_inflate     #
    #####################################

    echo "Inflating mid-thickness surfaces..."

    mris_inflate \
    -n 5 \
    -dist 0.15 \
    -no-save-sulc \
    lh.midthickness.32k.surf.gii \
    lh.midthickness.inflated.32k.surf.gii

    mris_inflate \
    -n 5 \
    -dist 0.15 \
    -no-save-sulc \
    rh.midthickness.32k.surf.gii \
    rh.midthickness.inflated.32k.surf.gii

    echo "All steps complete!"
done