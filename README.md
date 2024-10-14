# MYSPACE-Parcellation of ROI with fmri resting state
KEYPOINTS : fmri analysis, single subject, parcellation, resting state
## Intro
This project is about the parcellation of body maps using resting state fmri data. This provided code can be used on any part of the brain
### Inputs 
3 files :
- a fmri resting state data (4D): nii format
- a mask that highlight the cerebral cortex (3D) : nii format
- a ROI, in our case the S1 (3D): nii format
Note : the fmri data must have been preprocessed, slicetimng, coregistration, ,smoothing,ICA-FIX
### Outputs
- 
## Similarity matrix computation
Similarity map generated based on a Congrad github. Pearson coorelation of the voxels fingerprints within a ROI

## Spatial Gradient magnitude map
These map is created by looking at the neighbours of each voxel and comparing their similarity maps.