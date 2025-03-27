"""
Module Name: preprocessing_surface.py
Description:
    This script is used to load and process volume and surface data from Freesurfer outputs.
    It includes utilities for:
      - Loading and preprocessing fMRI volume data
      - Projecting fMRI volume data onto the cortical surface
      - Converting fMRI volume data to spatial modes using SVD

Author: Robin Junod, robin.junod@epfl.ch
Created: 2025-01-16

"""
#%%
import os
import numpy as np
import nibabel as nib
from datetime import datetime
from nilearn import surface
from nilearn.image import resample_img

def load_data_normalized(surf_fmri_path,
                         vol_fmri_path, 
                         brain_mask_path):
    """
    Load and normalize surface and volume fMRI data.

    Args:
        surf_fmri_path (str): Path to the surface fMRI data file.
        vol_fmri_path (str): Path to the volume fMRI data file.
        brain_mask_path (str): Path to the brain mask file.

    Returns:
        tuple: A tuple containing:
            - surf_fmri_n (numpy.ndarray): Normalized surface fMRI data.
            - vol_fmri_n (numpy.ndarray): Normalized volume fMRI data.
    """
    # Load the surface data
    surf_fmri = nib.load(str(surf_fmri_path)).get_fdata().squeeze()
    surf_fmri_n = (surf_fmri - np.mean(surf_fmri, axis=1, keepdims=True)) / np.std(surf_fmri, axis=1, keepdims=True)
    
    # Load the images using nibabel
    vol_fmri = nib.load(str(vol_fmri_path))
    mask_img = nib.load(brain_mask_path)

    # resample the mask to the right one
    mask_img = resample_img(
        mask_img,
        target_affine=vol_fmri.affine,
        target_shape=vol_fmri.get_fdata().shape[:-1],
        interpolation='nearest',
        force_resample=True
    )

    fmri_data = vol_fmri.get_fdata()
    mask_data = mask_img.get_fdata().astype(bool)  # Convert mask to boolean
    # Normalize the data
    vol_fmri = fmri_data[mask_data] 
    vol_fmri_n = (vol_fmri - np.mean(vol_fmri, axis=1, keepdims=True)) / np.std(vol_fmri, axis=1, keepdims=True)
    
    # Remove nans from the data
    surf_fmri_n = np.nan_to_num(surf_fmri_n).astype(np.float32)
    vol_fmri_n = np.nan_to_num(vol_fmri_n).astype(np.float32)
    return surf_fmri_n, vol_fmri_n # Warning : the vol outputs are not in 3D anymore


# This is a custom function for my data organisation
from typing import Literal
def extract_fmri_timeseries(dataset = Literal["PPSFACE_N18", "PPSFACE_N20"],
                            hemisphere = Literal["lh", "rh"],
                            run = Literal["1","2"]):
    # Load all of the surf fmri data
    surf_fmri_list = []
    dataset_dir = f"D:\Data_Conn_Preproc\\{dataset}"
    if dataset == "PPSFACE_N18":
        n = 19
    else:
        n = 21
    for i in range(1,n):
        if i == 5 and dataset=="PPSFACE_N20": # Subject 5 is missing
            continue
        subject = f"{i:02d}"
        subj_dir = dataset_dir + r"\sub-" + subject
        fmri_path = subj_dir + f"\\func\surf_conn_sub{subject}_run{run}_{hemisphere}.func.fsaverage6.mgh"
        surf_fmri_img = nib.load(fmri_path)
        surf_fmri = surf_fmri_img.get_fdata()
        surf_fmri = np.squeeze(surf_fmri) # just rearrange the MGH data
        # Normilize the data
        surf_fmri = (surf_fmri - np.mean(surf_fmri, axis=1, keepdims=True)) / np.std(surf_fmri, axis=1, keepdims=True)
        # Replace nan values with 0
        surf_fmri = np.nan_to_num(surf_fmri)
        surf_fmri_list.append(surf_fmri)
    return surf_fmri_list

def fmri_vol2surf(vol_fmri_img, path_midthickness_l, path_midthickness_r):
    """
    An alternative to the Freesurfer project command.
    Projects fMRI volume data onto the cortical surface. 

    Args:
        vol_fmri_img (nib.Nifti1Image): The fMRI volume image to be projected onto the surface.
        path_midthickness_l (str): File path to the left hemisphere midthickness surface.
        path_midthickness_r (str): File path to the right hemisphere midthickness surface.

    Returns:
        tuple: A tuple containing:
            - surf_fmri_l (numpy.ndarray): The fMRI data projected onto the left hemisphere surface.
            - surf_fmri_r (numpy.ndarray): The fMRI data projected onto the right hemisphere surface.
    """
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

def fmri_to_spatial_modes(vol_fmri, 
                          resampled_mask,
                          n_modes=380,
                          low_variance_threshold = 0.5):
    """
    Converts fMRI volume data to spatial modes using Singular Value Decomposition (SVD).

    Args:
        vol_fmri (numpy.ndarray float): 4D fMRI volume data.
        resampled_mask (numpy.ndarray bool): 3D resampled brain mask (bool).
        n_modes (int): Number of spatial modes to retain.
        low_variance_threshold (float): Threshold for removing voxels with low variance.

    Returns:
        numpy.ndarray: 2D array of spatial modes.
    """
    from scipy.linalg import svd
    mask_idx = np.where(resampled_mask)
    # Create the spatial modes
    fmri_masked = vol_fmri[mask_idx]
    # Remove voxels with low variance (thresholding by the variance of each voxel's time series)
    variance = np.var(fmri_masked, axis=1)
    fmri_masked = fmri_masked[variance > low_variance_threshold]
    
    fmri_m_normalized = (fmri_masked - np.mean(fmri_masked, axis=1, keepdims=True)) / np.std(fmri_masked, axis=1, keepdims=True)
    keep = ~np.isnan(fmri_m_normalized).any(axis=1) & ~np.isinf(fmri_m_normalized).any(axis=1)
    fmri_m_normalized = fmri_m_normalized[keep]
    # The principal components are the eigenvectors of S = X'*X./(n-1), but computed using SVD
    [U,sigma,V] = svd(fmri_m_normalized,full_matrices=False)
    # Project X onto the principal component axes
    # spatial_modes = U[:n_modes,:].T * sigma[:n_modes]
    
    return U[:n_modes,:]
    

def congrads_dim_reduction(vol_fmri, resampled_mask, low_variance_threshold=0.5):
    # A copy from congrads for comparison
    def pca(X):
        from scipy.linalg import svd

        # Center X by subtracting off column means
        X -= np.mean(X,0)

        # The principal components are the eigenvectors of S = X'*X./(n-1), but computed using SVD
        [U,sigma,V] = svd(X,full_matrices=False)

        # Project X onto the principal component axes
        Y = U*sigma

        # Convert the singular values to eigenvalues 
        sigma /= np.sqrt(X.shape[0]-1)
        evals = np.square(sigma)
        
        return V, Y, evals
    
    mask_idx = np.where(resampled_mask)
    # Create the spatial modes
    fmri_masked = vol_fmri[mask_idx]
    # Remove voxels with low variance (thresholding by the variance of each voxel's time series)
    variance = np.var(fmri_masked, axis=1)
    fmri_masked = fmri_masked[variance > low_variance_threshold]
    [evecs,Bhat,evals] = pca(fmri_masked)
    # PLEASE TELL ME THAT IT IS STUPID PLEASSEEEE
    
#%% Run the code
if __name__ == "__main__":
    pass
# %%
