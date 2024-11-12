# MYSPACE-Parcellation of ROI with fmri resting state
KEYPOINTS : fmri analysis, single subject, parcellation, resting state
## Intro
This project is about the parcellation of body maps using resting state fmri data. This provided code can be used on any part of the brain
### Preprocessing steps
The preprocessing steps is a very improtant step in order to get meaningful results. The tricky part is to denoise the data by using FIX-ICA. The easiest way for that is to use FSL (a well known software for fmri analysis). Here are the steps cnsidering that you have raw fmri and T1 scan of one subject.
- put you fmri file in a folder named 'func' and you T1 scan in a fodler named 'anat'
- Use Bet from FSL to extract the brain of your anat data. You will have now a file in 'anat' folder with that named ending with _brain.nii.gz

- Next use the Melodic GUI to perform basic preprocessing as well as the ICA. (look at Andy's brain book tutorial ICA). You just need to give the GUI the frmi 4D data path. And a path to the brain extracted T1 scan. Tune the other preprocessing steps or let them like that and then click the go button

- Use the command 'fix -m melodic_output.ica Standard 20'

### Inputs 
3 files :
- a fmri resting state data (4D): nii format
- a mask that highlight the cerebral cortex (3D) : nii format
- a ROI, in our case the S1 (3D): nii format
Note : the fmri data must have been preprocessed: brain extraction, motion correction, coregistration, noramlization, smoothing, ICA-FIX
### Outputs
- 
## Similarity matrix computation
Similarity map generated based on a Congrad github. Pearson coorelation of the voxels fingerprints within a ROI

## Spatial Gradient magnitude map
These map is created by looking at the neighbours of each voxel and comparing their similarity maps.