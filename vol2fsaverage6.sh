#!/bin/bash

# -------------------------------------------------------------------
# Script: project_and_map_fmri.sh
# Description: Automates the projection of fMRI data to the cortical
#              surface and maps the data to fsaverage6 space for
#              subjects sub02 to sub18.
# $ cd /mnt/d/DATA_min_preproc/dataset_study1
# $ export SUBJECTS_DIR=/home/rjunod/freesurfer/conn_PPSFACE_study2
# or $ export SUBJECTS_DIR=/home/rjunod/freesurfer/conn_PPSFACE_study2 # FREESURFER SUBJECTS DIRECTORY (where surf data is)



# How to use htis script 
# 1. Having freesurfer SUBJDIR with the ouputs of recon-all (e.g. /home/rjunod/freesurfer/conn_PPSFACE_study2)
# 2. Having the functional data in the BIDS (func/anat folders) format (e.g. /mnt/d/DATA_min_preproc/dataset_study1)
# 3. Put the script in the BIDS directory
# 4. Modify the script to match the paths
# 4. Run the script in the terminal (maybe use unix2dos to convert the file to unix format)


# -------------------------------
# 1. Define Subject Range
# -------------------------------

# Loop through subject numbers 02 to 20
for subj_num in {01..20}; do
  for run_num in {1..2}; do
    echo "----------------------------------------"
    echo "Processing Subject: sub-$subj_num"
    echo "----------------------------------------"
    
    # -------------------------------
    # 2. Define File Paths
    # -------------------------------


    # START OF MODIFICATION
    # Define subject-specific directories
    FUNC_DIR="sub-$subj_num/func"
    SUBJECT_FREESURFER_DIR="${SUBJECTS_DIR}/sub${subj_num}_freesurfer"
    
    # Define input functional MRI NIfTI file
    MOV_FILE="${FUNC_DIR}/niftiDATA_Subject${subj_num}_Condition000_run${run_num}.nii.gz"

    # Define registration header
    REGHEADER="sub${subj_num}_freesurfer" # looking at dir in $SUBJECTS_DIR
    
    # Define output filenames for left and right hemispheres
    OUTPUT_LH="${FUNC_DIR}/surf_conn_sub${subj_num}_run${run_num}_lh.func.mgh"
    OUTPUT_RH="${FUNC_DIR}/surf_conn_sub${subj_num}_run${run_num}_rh.func.mgh"
    
    # Define output filenames for fsaverage6 space
    OUTPUT_LH_FSAVERAGE="${FUNC_DIR}/surf_conn_sub${subj_num}_run${run_num}_lh.func.fsaverage6.mgh"
    OUTPUT_RH_FSAVERAGE="${FUNC_DIR}/surf_conn_sub${subj_num}_run${run_num}_rh.func.fsaverage6.mgh"
    
    # END OF MODIFICATION

    
    # -------------------------------
    # 3. Check if Functional Data Exists
    # -------------------------------
    
    if [ ! -f "$MOV_FILE" ]; then
        echo "Functional data file $MOV_FILE not found. Skipping sub-$subj_num."
        continue
    fi
    
    # -------------------------------
    # 4. Project fMRI Data to Left Hemisphere
    # -------------------------------
    
    echo "Projecting fMRI data to Left Hemisphere..."
    mri_vol2surf \
      --mov "$MOV_FILE" \
      --regheader "$REGHEADER" \
      --hemi lh \
      --surf white \
      --projfrac 0.5 \
      --interp trilinear \
      --o "$OUTPUT_LH"
    
    if [ $? -ne 0 ]; then
        echo "Error in mri_vol2surf for Left Hemisphere of sub-$subj_num. Skipping to next subject."
        continue
    fi
    
    # -------------------------------
    # 5. Project fMRI Data to Right Hemisphere
    # -------------------------------
    
    echo "Projecting fMRI data to Right Hemisphere..."
    mri_vol2surf \
      --mov "$MOV_FILE" \
      --regheader "$REGHEADER" \
      --hemi rh \
      --surf white \
      --projfrac 0.5 \
      --interp trilinear \
      --o "$OUTPUT_RH"
    
    if [ $? -ne 0 ]; then
        echo "Error in mri_vol2surf for Right Hemisphere of sub-$subj_num. Skipping to next subject."
        continue
    fi
    
    # -------------------------------
    # 6. Map Left Hemisphere Data to fsaverage6
    # -------------------------------
    
    echo "Mapping Left Hemisphere data to fsaverage6..."
    mri_surf2surf \
      --srcsubject "sub${subj_num}_freesurfer" \
      --trgsubject fsaverage6 \
      --hemi lh \
      --sval "$OUTPUT_LH" \
      --tval "$OUTPUT_LH_FSAVERAGE" \
      --surfreg sphere.reg
    
    if [ $? -ne 0 ]; then
        echo "Error in mri_surf2surf for Left Hemisphere of sub-$subj_num. Skipping to next subject."
        continue
    fi
    
    # -------------------------------
    # 7. Map Right Hemisphere Data to fsaverage6
    # -------------------------------
    
    echo "Mapping Right Hemisphere data to fsaverage6..."
    mri_surf2surf \
      --srcsubject "sub${subj_num}_freesurfer" \
      --trgsubject fsaverage6 \
      --hemi rh \
      --sval "$OUTPUT_RH" \
      --tval "$OUTPUT_RH_FSAVERAGE" \
      --surfreg sphere.reg
    
    if [ $? -ne 0 ]; then
        echo "Error in mri_surf2surf for Right Hemisphere of sub-$subj_num. Skipping to next subject."
        continue
    fi
    
    echo "Completed processing for sub-$subj_num."
done

echo "All subjects processed."