#%%
import os
import sys
import scipy
import numpy as np
from toolbox_parcellation import pca, corr, eta2


def fingerprint_simmatrix_in_ROI(fmri_data: np.ndarray,
                                    roi_data: np.ndarray,
                                    mask_data: np.ndarray,
                                    save_simmatrix=True) -> np.ndarray:
    
    """Compute the similarity matrix for fMRI data based on connectivity fingerprints"""
    #TODO : VERIFY IF THIS APPROCHES ONLY MAKES ONE CORRELATION MATRIX OR DOES IT MAKE THE 
    # CORRELATION MATRIX OF THE CORRELATION MATRIX TO GET MORE CONECTIVITY INFORMATIONS.
	# Store the dimensions of the roi data and fmri data for later use
    roidims = roi_data.shape
    nVoxels = np.prod(roidims)
    nFrames = fmri_data.shape[3]
    
    # Reshape roi into a vector of size nVoxels
    mask_flatten = np.reshape(mask_data,(nVoxels))
    # Check if roi inside of mask
    roi_flatten = np.reshape(roi_data,(nVoxels))
    # Reshape fmri data into a matrix of size nVoxels x n
    fmri_flatten = np.reshape(fmri_data,(nVoxels,nFrames))
    fmri_flatten = fmri_flatten.astype(np.float32)
    
    # Verfiy that the data is good
    if (roi_flatten != roi_flatten * mask_flatten).any(): # Verify the ROI is inside the mask
        sys.exit('ROI is not inside the mask. Exiting. (use roi_flatten = roi_flatten * mask_flatten)')
    if (roi_flatten != roi_flatten * (np.var(fmri_flatten, axis=1) > 0)).any(): # Verify the ROI has positive variance
        sys.exit('ROI contains voxels without variance. Exiting. (use roi_flatten = roi_flatten * (np.var(fmri_flatten, axis=1) > 0)')
        
    # Find the indices inside roi
    roiIndices1D = np.where(roi_flatten > 0)
    spatial_position = np.where(roi_data > 0)
    # Find the indices outside roi but inside mask
    maskIndices = np.where((roi_flatten==0) & (mask_flatten>0)) 

    # Initialise similarity matrix
    S = np.zeros([np.sum(roi_flatten>0),np.sum(roi_flatten>0)])
    
    # Normalise the data
    fmri_normalized = (fmri_flatten - np.mean(fmri_flatten, axis=1, keepdims=True)) / np.std(fmri_flatten, axis=1, keepdims=True)
    
    # Gather data inside roi
    A = fmri_normalized[roiIndices1D,:][0]
    
    # If the roi contains invalid data it must be due to a division by 0 (stdev)
    # since the data themselves do not contain nans or infs. If so, we terminate 
    # the program and the user should define a roi covering functional data.
    # TODO : this part is now automatically taken care of by removing voxels
    # with 0 variance in the roi with the preprocesing func.
    if np.any(np.isnan(A)) or np.any(np.isinf(A)):
        print('WARNING : ROI includes ',
        np.isinf(np.var(A, axis=1)).sum() + np.isnan(np.var(A, axis=1)).sum(),
        ' voxels without variance.\nExiting.')
        
    # Gather data outside roi
    B = fmri_normalized[maskIndices,:][0]

    # Transpose so that the data are stored in a time x space matrix
    A = A.T
    B = B.T

    # A division by 0 (stdev) can also lead to nans and infs in the mask data.
    # In this case we can simply throw a warning and ignore all voxels without
    # variance.
    keepB = ~np.isnan(B).any(axis=0) & ~np.isinf(B).any(axis=0)
    if np.any(np.isnan(B)) or np.any(np.isinf(B)):
        print('WARNING: Mask includes voxels without variance.')
    # Delete data to free up memory
    del fmri_normalized  
    del fmri_flatten
    
    # Get voxel-wise connectivity fingerprints 
    print('Computing voxel-wise connectivity fingerprints...')
    [evecs,Bhat,evals] = pca(B[:,keepB])
    # Compute the correlation matrix of the ROI data with the connectivity fingerprints
    R = corr(A,Bhat)
    print('Computing similarity matrix...')
    S += eta2(R)
    print('Done.')
    return S, spatial_position


def simple_simmatrix_in_ROI(fmri_data: np.ndarray,
                            roi_data: np.ndarray) -> np.ndarray:
    """Compute a simple similarity matrix for a ROI.
    This similarity matrix is computed by correlating the fMRI data in the ROI.
    Args:
        fmri_data (np.ndarray): 4D fmri data
        roi_data (np.ndarray): 3D roi data
        mask_data (np.ndarray): 3D mask data
    Returns:
        np.ndarray: Similarity matrix
    """
    
    # Store the dimensions of the ROI data and fMRI data
    roidims = roi_data.shape
    nVoxels = np.prod(roidims)
    nFrames = fmri_data.shape[3]

    # Flatten the mask and ROI data
    print('Flattening the mask, ROI, and fMRI data...')
    roi_flatten = roi_data.flatten()
    spatial_position = np.where(roi_data > 0)
    roi_fmri = fmri_data[spatial_position]

    # Verify that the ROI has positive variance
    roi_frmi_flatten = roi_fmri.reshape(-1, nFrames)
    voxel_variances = np.var(roi_frmi_flatten, axis=1)
    
    if np.any(roi_frmi_flatten[voxel_variances > 0]==0):
        #TODO : create an algo to take care of this
        print('WARNING BIG ERROR ROI invalid, 0 variance for some voxels, automatic removing from the ROI.')
        roi_frmi_flatten = roi_flatten * (np.var(roi_frmi_flatten, axis=1) > 0) # Remove voxels with 0 variance from the ROI
        spatial_position = np.where((roi_frmi_flatten > 0).reshape(roidims))


    # Z-score normalization of the time series for each voxel
    roi_fmri_mean = np.mean(roi_frmi_flatten, axis=1, keepdims=True)
    roi_fmri_std = np.std(roi_frmi_flatten, axis=1, keepdims=True)
    roi_fmri_data_z = (roi_frmi_flatten - roi_fmri_mean) / roi_fmri_std

    # Handle any NaNs or Infs resulting from zero variance
    roi_fmri_data_z = np.nan_to_num(roi_fmri_data_z)

    # Compute the similarity matrix (correlation matrix)
    print('Compute similarity matrix in ROI')
    corr_mtrx = np.corrcoef(roi_fmri_data_z) # TODO : use the corr btwn voxel inside and ouside the roi
    # Similary matrix is a correlation matrix of the correlation matrix
    sim_mtrx = np.corrcoef(corr_mtrx)
    
    return sim_mtrx, spatial_position


def RSFC_singlevoxel():
    pass

#%%
if __name__=='__main__':

   # load a similarity matrix
    outdir_sim_mtrx = 'G:/HCP/outputs/sim_mtrx'
    out_base_name = 'similarity_matrix_S02_20241023_114059'
    file_path = os.path.join(outdir_sim_mtrx, out_base_name )
    loaded_data = scipy.io.loadmat(file_path)
    sim_matrix = loaded_data['S']
    spatial_position = loaded_data['spatial_position'] 
# %%
