#%% Preprocessing o fthe surface data

"""
Inputs:
- Surface data (gifti format) from freesurfer
- preprocessed fMRI data
- normalized T1 image

First convert to .gii
Then create the midthickness surface with mris_expand
Then reduce the number of vertex with mris_remesh

"""
#%%
import numpy as np
import nibabel as nib
from nilearn import surface, plotting
from scipy.ndimage import gaussian_filter
from nilearn.image import resample_img

# Paths
SUBJECT = r"01"

subj_dir = r"D:\DATA_min_preproc\dataset_study2\sub-" + SUBJECT
path_func = subj_dir + r"\func\rwsraOB_TD_FBI_S" + SUBJECT + r"_007_Rest.nii"

path_midthickness_r = subj_dir + r"\func\rh.midthickness.32k.surf.gii"
path_midthickness_l = subj_dir + r"\func\lh.midthickness.32k.surf.gii"

path_white_r = subj_dir + r"\func\rh.white.32k.surf.gii"
path_white_l = subj_dir + r"\func\lh.white.32k.surf.gii"

path_pial_r = subj_dir + r"\func\rh.pial.32k.surf.gii"
path_pial_l = subj_dir + r"\func\lh.pial.32k.surf.gii"

path_brain_mask = subj_dir + r"\sub" + SUBJECT + r"_freesurfer\mri\brainmask.mgz"


def preprocess_volume_data(path_func, path_brain_mask):
    """
    Preprocess the volume data
    """
    # Load volumetric data
    fmri_img = nib.load(path_func)
    affine_vol_fmri = fmri_img.affine
    vol_fmri = fmri_img.get_fdata()

    # Load mask data (and resample to target shape)
    brain_mask_img = nib.load(path_brain_mask)
    resampled_mask_img = resample_img(
        brain_mask_img,
        target_affine=affine_vol_fmri,
        target_shape=(79, 95, 79),
        interpolation='nearest'
    )
    # Convert to bool
    resampled_mask = resampled_mask_img.get_fdata()
    resampled_mask = (resampled_mask != 0) # Convert to bool
    resampled_mask_img = nib.Nifti1Image(resampled_mask, 
                                        affine=resampled_mask_img.affine, 
                                        header=resampled_mask_img.header)

    # Save the resampled mask
    resampled_mask_img.to_filename(subj_dir + r"\anat\resampled_brain_mask.nii.gz")

    # Normalize the time series of the volume data inside the mask
    vol_fmri_masked_ = vol_fmri[resampled_mask]
    vol_fmri_masked = np.zeros_like(vol_fmri)
    vol_fmri_masked[resampled_mask] = vol_fmri_masked_
    vol_fmri_img = nib.Nifti1Image(vol_fmri_masked, 
                                    affine=fmri_img.affine, 
                                    header=fmri_img.header)
    # #%% Visualize the mask and the fmri data
    # mean_fmri = np.mean(vol_fmri, axis=-1)
    # mean_fmri_img = nib.Nifti1Image(mean_fmri, affine_vol_fmri)
    # plotting.view_img(resampled_mask_img, 
    #                 bg_img=mean_fmri_img)
    
    return vol_fmri_img, resampled_mask_img




def fmri_vol2surf(vol_fmri_img, path_midthickness_l, path_midthickness_r):
    """ 
    Get the fmri data into the surface
    """
    # TODO : verify that the surface data has no NaN values
    surf_fmri_l = surface.vol_to_surf(vol_fmri_img,
                                      path_midthickness_l,
                                      radius=6)
    surf_fmri_r = surface.vol_to_surf(vol_fmri_img,
                                      path_midthickness_r,
                                      radius=6)
    
    if np.isnan(surf_fmri_l).any() or np.isnan(surf_fmri_r).any():
        print("NaN values in the surface data")
    
    # Normalized the time series of the surface data
    surf_fmri_l = (surf_fmri_l - np.mean(surf_fmri_l, axis=1, keepdims=True)) / np.std(surf_fmri_l, axis=1, keepdims=True)
    surf_fmri_r = (surf_fmri_r - np.mean(surf_fmri_r, axis=1, keepdims=True)) / np.std(surf_fmri_r, axis=1, keepdims=True)
    
    return surf_fmri_l, surf_fmri_r



#%% Run the code
if __name__ == "__main__":

    vol_fmri_img, resampled_mask_img = preprocess_volume_data(path_func, path_brain_mask)

    surf_fmri_l, surf_fmri_r = fmri_vol2surf(vol_fmri_img, path_midthickness_l, path_midthickness_r)

                                         
                                        


