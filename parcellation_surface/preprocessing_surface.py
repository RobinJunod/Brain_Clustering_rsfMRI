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
import os
import numpy as np
import nibabel as nib
from datetime import datetime
from nilearn import surface
from nilearn.image import resample_img

# Paths
SUBJECT = r"01"

SUBJ_DIR = r"D:\DATA_min_preproc\dataset_study2\sub-" + SUBJECT
path_func = SUBJ_DIR + r"\func\rwsraOB_TD_FBI_S" + SUBJECT + r"_007_Rest.nii"

path_midthickness_r = SUBJ_DIR + r"\func\rh.midthickness.32k.surf.gii"
path_midthickness_l = SUBJ_DIR + r"\func\lh.midthickness.32k.surf.gii"

path_white_r = SUBJ_DIR + r"\func\rh.white.32k.surf.gii"
path_white_l = SUBJ_DIR + r"\func\lh.white.32k.surf.gii"

path_pial_r = SUBJ_DIR + r"\func\rh.pial.32k.surf.gii"
path_pial_l = SUBJ_DIR + r"\func\lh.pial.32k.surf.gii"

path_brain_mask = SUBJ_DIR + r"\sub" + SUBJECT + r"_freesurfer\mri\brainmask.mgz"


def load_volume_data(path_func, path_brain_mask):
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
    resampled_mask_img.to_filename(SUBJ_DIR + r"\anat\resampled_brain_mask.nii.gz")

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
    
    return vol_fmri_img, resampled_mask_img, affine_vol_fmri


def downsample_volume_fmri(vol_fmri_img,
                           resampled_mask_img):
    """This part is made to reduce the number of vertex of the volume data.
    It will select the most important vertex of the volume data.
    Args:
        vol_fmri_img (_type_): _description_
    """
    vol_fmri = vol_fmri_img.get_fdata()
    # Create a mask 
    # Volume spatial smoothing nibabel
    # vol_fmri_img_smooth = (vol_fmri_img, fwhm=6)
    # Remove the vortex with a low signal to noise ratio
    
    # Select few vortex within the volume data
    
    return None


def fmri_to_spatial_modes(vol_fmri_img, 
                          resampled_mask_img,
                          n_modes=10_000):
    """
    Converts fMRI volume images to spatial modes using Singular Value Decomposition (SVD).

    Args:
        vol_fmri_img (Nifti1Image): 4D fMRI volume image.
        resampled_mask_img (Nifti1Image): 3D resampled mask image.

    Returns:
        numpy.ndarray: 2D array of spatial modes.
    """
    from scipy.linalg import svd
    mask = resampled_mask_img.get_fdata()
    mask_idx = np.where(mask)
    fmri = vol_fmri_img.get_fdata()
    # Create the spatial modes
    fmri_masked = fmri[mask_idx]
    # keep the voxel with more 50% of the variance 
    
    fmri_m_normalized = (fmri_masked - np.mean(fmri_masked, axis=1, keepdims=True)) / np.std(fmri_masked, axis=1, keepdims=True)
    keep = ~np.isnan(fmri_m_normalized).any(axis=1) & ~np.isinf(fmri_m_normalized).any(axis=1)
    fmri_m_normalized = fmri_m_normalized[keep]
    # The principal components are the eigenvectors of S = X'*X./(n-1), but computed using SVD
    [U,sigma,V] = svd(fmri_m_normalized,full_matrices=False)
    # Project X onto the principal component axes
    # spatial_modes = U[:n_modes,:].T * sigma[:n_modes]
    
    return U[:n_modes,:]
    
    
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


def save_similartiy_matrix(similarity_matrix, output_dir):
    """
    Save the similarity matrix (WARNING : not really convinient to save this matrix)
    """
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    path = output_dir + f"\similarity_matrix_{time}.npy"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, similarity_matrix)

def load_similarity_matrix(path):
    """
    Load the similarity matrix
    """
    similarity_matrix = np.load(path)
    return similarity_matrix


def save_gradient_map(gradient_map, output_dir):
    """
    Save the gradient map
    """
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    path = output_dir + f"\gradient_map_{time}.npy"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, gradient_map)
    
def load_gradient_map(path):
    """
    Load the gradient map
    """
    gradient_map = np.load(path)
    return gradient_map





#%% Run the code
if __name__ == "__main__":

    vol_fmri_img, resampled_mask_img, affine = load_volume_data(path_func, path_brain_mask)

    surf_fmri_l, surf_fmri_r = fmri_vol2surf(vol_fmri_img, path_midthickness_l, path_midthickness_r)

                                         
                                        



# %%
