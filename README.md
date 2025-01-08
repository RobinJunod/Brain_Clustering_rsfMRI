# MYSPACE- RSFC Parcellation with fMRI resting state
KEYPOINTS : fmri analysis, single subject, parcellation, resting state
## Intro

This project dives into the parcellation of the brain using resting state functional connectivity (RSFC). It is based on multiples approches for parcellation and is made of full python pipelines. 

Before using these scripts some preprocessings steps must be done.

---
### Preprocessing steps
For the preprocessing steps you should use the following softwares:
- Freesurfer 
- SPM (or FSL)

Some type of denoising such as ICA or using GLM to remove artefacts will improve the results and lead to more significant parcellation.

The tricky part is to denoise the data by using FIX-ICA. The easiest way for that is to use FSL (a well known software for fmri analysis). Here are the steps cnsidering that you have raw fmri and T1 scan of one subject.
- put you fmri file in a folder named 'func' and you T1 scan in a fodler named 'anat'
- Use Bet from FSL to extract the brain of your anat data. You will have now a file in 'anat' folder with that named ending with _brain.nii.gz
- Next use the Melodic GUI to perform basic preprocessing as well as the ICA. (look at Andy's brain book tutorial ICA). You just need to give the GUI the frmi 4D data path. And a path to the brain extracted T1 scan. Tune the other preprocessing steps or let them like that and then click the go button
- Use the command 'fix -m melodic_output.ica Standard 20'

---

## Volume based parcellation
For more details look at the *readme* inside of the folder 'parcellation_volume'.
### Inputs
- a fmri resting state data (4D): nii format
- a mask that highlight the cerebral cortex (3D) : nii format
- a ROI, in our case the S1 (3D): nii format
Note : the fmri data must have been preprocessed: brain extraction, motion correction, coregistration, noramlization, smoothing, ICA-FIX

## Surface based parcellation
### Prerequest
- Freesurfer recon-all
- Resting state fMRI : preprocessed (brain extraction, motion correction, coregistration, noramlization, smoothing)
