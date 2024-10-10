
#%%
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import nibabel as nib

from toolbox_parcellation import extract_4Ddata_from_nii, extract_3Ddata_from_nii


def preprocess_ROI_data(roi_data: np.ndarray,
                        mask_data: np.ndarray,
                        fmri_data: np.ndarray) -> np.ndarray:
    """Preprocess the ROI data"""
    # Keep only the ROI voxels inside the mask
    roi_data = roi_data * mask_data
    # Keep only the ROI voxels that have positive variace in the fMRI data
    voxel_variances = np.var(fmri_data, axis=3)
    roi_data = roi_data * (voxel_variances > 0)
    return roi_data


def compute_similarity_matrix_in_ROI(fmri_data: np.ndarray,
                                    roi_data: np.ndarray,
                                    mask_data: np.ndarray,
                                    save_simmatrix=True) -> np.ndarray:
    
    """Compute the similarity matrix for fMRI data"""
    
    from toolbox_parcellation import pca, corr, eta2
    
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
    
    # Verfiy that the data is good
    if roi_flatten != roi_flatten * mask_flatten: # Verify the ROI is inside the mask
        sys.exit('ROI is not inside the mask. Exiting. (use roi_flatten = roi_flatten * mask_flatten)')
    if roi_flatten != roi_flatten * (np.var(fmri_flatten, axis=1) > 0): # Verify the ROI has positive variance
        sys.exit('ROI contains voxels without variance. Exiting. (use roi_flatten = roi_flatten * (np.var(fmri_flatten, axis=1) > 0)')
        
    # Find the indices inside roi
    roiIndices1D = np.where(roi_flatten > 0)
    SpatialPositions = np.where(roi_data > 0)
    # Find the indices outside roi but inside mask
    maskIndices = np.where((roi_flatten==0) & (mask_flatten>0)) 

    # Initialise similarity matrix
    S = np.zeros([np.sum(roi_flatten>0),np.sum(roi_flatten>0)])

    data = fmri_flatten - np.tile(np.mean(fmri_flatten,1),(nFrames,1)).T
    data = data / np.tile(np.std(data,1),(nFrames,1)).T
    
    # Gather data inside roi
    A = data[roiIndices1D,:][0]
    
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
    B = data[maskIndices,:][0]

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
    del data  
    del fmri_flatten
    
    # Get voxel-wise connectivity fingerprints 
    print('Computing voxel-wise connectivity fingerprints...')
    [evecs,Bhat,evals] = pca(B[:,keepB])
    # Compute the correlation matrix of the ROI data with the connectivity fingerprints
    R = corr(A,Bhat)
    print('Computing similarity matrix...')
    S += eta2(R)
    
    # If requested, save the similarity matrix as a matlab .mat file
    if save_simmatrix:
        import scipy.io
        outdir = 'G:/RSFC/output_folder'
        out_base_name = 'similarity_matrix_S02'
        scipy.io.savemat(outdir + "/" +  out_base_name + ".eta2", dict(S=S))
    
    return S, SpatialPositions



def compute_gradient_embedding(similarity_matrix, n_components=3, alpha=0.5):
    from scipy.linalg import eigh
    from scipy.sparse.linalg import eigsh
    """
    Compute the gradient map from a similarity matrix using diffusion embedding.

    Parameters:
    - similarity_matrix: (N x N numpy array) Similarity matrix between ROIs.
    - n_components: (int) Number of gradient components to compute.
    - alpha: (float) Normalization parameter in diffusion maps (0 <= alpha <= 1).

    Returns:
    - gradients: (N x n_components numpy array) Gradient map.
    """
    # Ensure the similarity matrix is symmetric and non-negative
    S = np.maximum(similarity_matrix, 0)
    S = (S + S.T) / 2.0
    # Compute the degree matrix D
    degrees = np.sum(S, axis=1)
    D = np.diag(degrees)
    # Normalize the affinity matrix
    D_alpha = np.diag(degrees ** (-alpha))
    K = D_alpha @ S @ D_alpha
    # Compute the transition matrix M
    degrees_K = np.sum(K, axis=1)
    D_K_inv = np.diag(1.0 / (degrees_K + 1e-10))
    M = D_K_inv @ K
    # Perform eigendecomposition
    eigenvalues, eigenvectors = eigsh(M, k=n_components+1, which='LM')
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Exclude the first eigenvector (trivial solution)
    gradients = eigenvectors[:, 1:n_components+1]
    return gradients





# Compute Similarity Matrix inside a ROI
def compute_gradient_map_in_ROI(SimilarityMatrix : np.ndarray,
                                roi_data :  np.ndarray,) -> np.ndarray:
    # put the similarity vectors back in the 3D space
    # look for neigbours in the 3D space
    # compute the gradient map
    pass


# 3. Apply Watershed Algorithm for Parcellation
def apply_watershed(similarity_matrix,
                    fmri_data) -> np.ndarray:
    pass













if __name__ == '__main__':
    path_fmri = 'G:/RSFC/resting_state_data/rwsraOB_TD_FBI_S01_007_Rest.nii'
    path_roi = 'G:/RSFC/ROI_data/rROI_postcentral.nii'
    path_mask = 'G:/RSFC/ROI_data/rMASK_wholebrain.nii'
    
    fmri_data = extract_4Ddata_from_nii(path_fmri)
    roi_data = extract_3Ddata_from_nii(path_roi)
    mask_data = extract_3Ddata_from_nii(path_mask)
    # Check the dimensions of the data
    if not fmri_data.shape[:-1] == mask_data.shape == roi_data.shape:
        sys.exit('Data dimensions do not match. Exiting.')
    # Preprocess the ROI data
    roi_data = preprocess_ROI_data(roi_data, mask_data, fmri_data)    
    nVoxels = np.prod(roi_data.shape)
    # Compute the similarity matrix
    compute_sim_matrix=False
    if compute_sim_matrix:
        sim_matrix = compute_similarity_matrix_in_ROI(fmri_data, roi_data, mask_data)
    else:
        # load the similarity matrix
        file_path_S = 'G:/RSFC/output_folder/similarity_matrix_S02.eta2'
        S = scipy.io.loadmat(file_path_S)['S']

    
    # compute the Gradient map
    nVoxels = np.prod(roi_data.shape)
    roi_flatten = np.reshape(roi_data,(nVoxels)) 
    roiIndices = np.where(roi_flatten > 0)
    # Initialise similarity matrix
    sim_map_matrix = np.zeros([nVoxels, len(S)])
    sim_map_matrix[roiIndices,:] = S
    # reshape the similarity in the 4D space
    sim_map_matrix = np.reshape(sim_map_matrix, roi_data.shape + (S.shape[1],))
    
    # Initialize the gradient magnitude array
    gradient_magnitude = np.zeros(sim_map_matrix.shape[:3])
    
    
    # Find the indices of voxels with non-zero vectors
    non_zero_voxel_indices = np.nonzero(np.any(sim_map_matrix != 0, axis=3))
    # Unpack the indices
    x_indices, y_indices, z_indices = non_zero_voxel_indices
    # Initialize the gradient magnitude array
    gradient_magnitude = np.zeros(sim_map_matrix.shape[:3])  # Shape (79, 95, 79)
    # Get the shape of the spatial dimensions
    x_max, y_max, z_max = sim_map_matrix.shape[:3]
    # For each voxel with a non-zero vector
    # for x, y, z in zip(x_indices, y_indices, z_indices):
    #     delta_x2 = delta_y2 = delta_z2 = 0.0
    #     # Check for neighboring voxels in the x-direction
    #     # Positive x-direction
    #     if x + 1 < x_max and np.any(sim_map_matrix[x + 1, y, z, :] != 0):
    #         delta = sim_map_matrix[x + 1, y, z, :] - sim_map_matrix[x, y, z, :]
    #         delta_x2 = np.dot(delta, delta)
    #     # Negative x-direction
    #     elif x - 1 >= 0 and np.any(sim_map_matrix[x - 1, y, z, :] != 0):
    #         delta = sim_map_matrix[x, y, z, :] - sim_map_matrix[x - 1, y, z, :]
    #         delta_x2 = np.dot(delta, delta)
    #     # Check for neighboring voxels in the y-direction
    #     # Positive y-direction
    #     if y + 1 < y_max and np.any(sim_map_matrix[x, y + 1, z, :] != 0):
    #         delta = sim_map_matrix[x, y + 1, z, :] - sim_map_matrix[x, y, z, :]
    #         delta_y2 = np.dot(delta, delta)
    #     # Negative y-direction
    #     elif y - 1 >= 0 and np.any(sim_map_matrix[x, y - 1, z, :] != 0):
    #         delta = sim_map_matrix[x, y, z, :] - sim_map_matrix[x, y - 1, z, :]
    #         delta_y2 = np.dot(delta, delta)
    #     # Check for neighboring voxels in the z-direction
    #     # Positive z-direction
    #     if z + 1 < z_max and np.any(sim_map_matrix[x, y, z + 1, :] != 0):
    #         delta = sim_map_matrix[x, y, z + 1, :] - sim_map_matrix[x, y, z, :]
    #         delta_z2 = np.dot(delta, delta)
    #     # Negative z-direction
    #     elif z - 1 >= 0 and np.any(sim_map_matrix[x, y, z - 1, :] != 0):
    #         delta = sim_map_matrix[x, y, z, :] - sim_map_matrix[x, y, z - 1, :]
    #         delta_z2 = np.dot(delta, delta)
    #     # Compute the gradient magnitude at the voxel
    #     gradient_magnitude[x, y, z] = np.sqrt(delta_x2 + delta_y2 + delta_z2)


    # Compute the gradient map
    # n_components = 3
    # alpha = 0.5
    # gradients = compute_gradient_embedding(S, n_components, alpha)
    

# %%
def compute_gradient_map(sim_matrix,
                         SpatialPositions,
                         roi_data_shape,
                         save_map: bool=True) -> np.ndarray:
    """Compute the gradient map from a similarity matrix using diffusion embedding.
    the gradient map is a 3D map of brain highligting place with similar connectivity.
    
    Args:
        sim_matrix (np.array): Computed from the similarity matrix
        SpatialPositions (tuple): Given by the function compute_similarity_matrix_in_ROI
        roi_data_shape (np.array 3D): Shape of the ROI data

    Returns:
        np.ndarray: Gradient map
    """
    # Unpack the spatial positions
    n_voxels = sim_matrix.shape[0]
    x_coords, y_coords, z_coords = SpatialPositions
    coord_to_index = {
        (x_coords[i], y_coords[i], z_coords[i]): i for i in range(len(x_coords))
    }
    # initialize the gradient magnitude array
    gradient_magnitude_map = np.zeros(roi_data_shape)
    # search for neighbours in the 3D space
    for i in range(n_voxels):
        x, y, z = x_coords[i], y_coords[i], z_coords[i]
        if sim_matrix[i].all() == 0:
            continue
        num_neighbours = 0
        rmse = 0
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            if (new_x, new_y, new_z) in coord_to_index:
                j = coord_to_index[(new_x, new_y, new_z)]
                rmse += np.linalg.norm(sim_matrix[i] - sim_matrix[j])
                num_neighbours += 1
        if num_neighbours > 0:
            gradient_magnitude_map[x, y, z] = rmse / num_neighbours
    
    # Save the gradient magnitude as a .nii file
    if save_map == True:
        import os
        import nibabel as nib
        from datetime import datetime
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(script_dir, 'output_folder/grad_maps')
        out_base_name = f'gradient_map{datetime.now().strftime('%Y%m%d_%H%M%S')}'
        # Ensure the output directory exists
        os.makedirs(outdir, exist_ok=True)
        output_path = os.path.join(outdir, out_base_name + '.nii')
        # Create a NIfTI image using nibabel
        nii_img = nib.Nifti1Image(gradient_magnitude_map, affine=np.eye(4))
        # Save the NIfTI image to the output path
        nib.save(nii_img, output_path)
    return gradient_magnitude_map 
                