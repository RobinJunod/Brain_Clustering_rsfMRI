
#%%

import sys
import os
from datetime import datetime
import numpy as np
import nibabel as nib
import scipy.io
import matplotlib.pyplot as plt

from toolbox_parcellation import extract_4Ddata_from_nii, extract_3Ddata_from_nii
from toolbox_parcellation import pca, corr, eta2



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
    print('Done.')
    return S, spatial_position


def compute_gradient_map(sim_matrix,
                         spatial_position,
                         roi_data_shape) -> np.ndarray:
    """Compute the gradient map from a similarity matrix using diffusion embedding.
    the gradient map is a 3D map of brain highligting place with similar connectivity.
    
    Args:
        sim_matrix (np.array): Computed from the similarity matrix
        spatial_position (tuple): Given by the function compute_similarity_matrix_in_ROI
        roi_data_shape (3D np.array): Shape of the ROI data

    Returns:
        np.ndarray: Gradient magnitude map
    """
    print('Computing gradient map...')
    # Unpack the spatial positions
    n_voxels = sim_matrix.shape[0]
    x_coords, y_coords, z_coords = spatial_position
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
    return gradient_magnitude_map 

def compute_gradient_map_gaussian(sim_matrix,
                                  spatial_position,
                                  roi_data_shape,
                                  sigma=1.0,
                                  radius=2) -> np.ndarray:
    """Compute the gradient map from a similarity matrix using a Gaussian-weighted neighborhood.
    The gradient map is a 3D map highlighting places with similar connectivity.

    Args:
        sim_matrix (np.array): Computed from the similarity matrix.
        spatial_position (tuple): Given by the function compute_similarity_matrix_in_ROI.
        roi_data_shape (tuple): Shape of the ROI data (should be 3D).
        sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
        radius (int, optional): Radius of the neighborhood to consider. Defaults to 3.

    Returns:
        np.ndarray: Gradient magnitude map.
    """
    print('Computing gradient map with Gaussian-weighted neighborhood...')
    # Unpack the spatial positions
    n_voxels = sim_matrix.shape[0]
    x_coords, y_coords, z_coords = spatial_position
    coord_to_index = {
        (x_coords[i], y_coords[i], z_coords[i]): i for i in range(n_voxels)
    }
    # Initialize the gradient magnitude array
    gradient_magnitude_map = np.zeros(roi_data_shape)
    # Precompute the possible offsets and their Gaussian weights
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the center voxel
                distance_squared = dx**2 + dy**2 + dz**2
                weight = np.exp(-distance_squared / (2 * sigma**2))
                if weight > 1e-6:  # Ignore negligible weights
                    offsets.append((dx, dy, dz, weight))
    # Compute the gradient magnitude map
    for idx in range(n_voxels):
        x_i, y_i, z_i = x_coords[idx], y_coords[idx], z_coords[idx]
        if not np.any(sim_matrix[idx]):
            continue  # Skip if the similarity vector is all zeros
        weighted_rmse = 0.0
        total_weight = 0.0
        for dx, dy, dz, weight in offsets:
            x_j, y_j, z_j = x_i + dx, y_i + dy, z_i + dz
            # Check if the neighbor voxel is within the ROI
            if (x_j, y_j, z_j) in coord_to_index:
                jdx = coord_to_index[(x_j, y_j, z_j)]
                diff = np.linalg.norm(sim_matrix[idx] - sim_matrix[jdx])
                weighted_rmse += weight * diff
                total_weight += weight
        if total_weight > 0:
            gradient_magnitude_map[x_i, y_i, z_i] = weighted_rmse / total_weight
    
    return gradient_magnitude_map


def blur_gradient_map(gradient_magnitude_map: np.ndarray,
                      sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to the gradient magnitude map."""
    from scipy.ndimage import gaussian_filter
    print('Applying Gaussian blur to the gradient magnitude map...')
    blurred_map = gaussian_filter(gradient_magnitude_map, sigma=sigma)
    return blurred_map


def find_seeds(gradient_magnitude_map: np.ndarray) -> np.ndarray:
    """Find the seeds for the watershed algorithm.
    Args:
        gradient_magnitude_map (3D np.ndarray): Gradient magnitude map
    Returns:
        np.ndarray: Seeds for the watershed algorithm
    """
    spatial_position = np.where(gradient_magnitude_map > 0)
    n_voxels = len(spatial_position[0])
    
    
    x_coords, y_coords, z_coords = spatial_position
    coord_to_index = {
        (x_coords[i], y_coords[i], z_coords[i]): i for i in range(len(x_coords))
    }
    # intialize the seeds map (0 for no seed, 1 for seed)
    seeds_maps = np.zeros(gradient_magnitude_map.shape)
    # search for the seeds in the 3D space
    for i in range(n_voxels):
        x, y, z = x_coords[i], y_coords[i], z_coords[i]
        n_higher_grad_neighbours = 0
        # Generate all neighbors in the 3x3x3 cube, including the center voxel itself
        neighbors = [(x + dx, y + dy, z + dz)
                    for dx in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    for dz in [-1, 0, 1]]
        neighbors.remove((x, y, z))
        for new_x, new_y, new_z in neighbors:
            # check if the neighbour has smaller value
            if (new_x, new_y, new_z) in coord_to_index:
                if gradient_magnitude_map[new_x, new_y, new_z] < gradient_magnitude_map[x, y, z]:
                    break
            n_higher_grad_neighbours += 1
        if n_higher_grad_neighbours > 25:
            seeds_maps[x, y, z] = 1 # seed 
               
    return seeds_maps


# TODO : Implement the watershed algorithm BIG TODO (for now it doesn't work)
import heapq
from scipy import ndimage

def watershed_by_flooding(gradient_magnitude_map: np.ndarray, 
                          seeds_maps: np.ndarray) -> np.ndarray:

    print('Performing watershed by flooding...')
    # gradient map preprocessing
    indices = np.where(gradient_magnitude_map==0)
    gradient_magnitude_map[indices] = np.inf
    # create mask
    mask = np.ones(gradient_magnitude_map.shape)
    mask[indices] = 0
    
    # Initialize labels array
    labels, num_labels = ndimage.label(seeds_maps > 0)
    
    # Initialize priority queue
    # Elements are tuples: (priority, x, y, z, label)
    heap = []
    
    # Create a status array to keep track of pixels
    # status == 0: unprocessed, status == 1: in queue, status == 2: processed
    status = np.zeros(labels.shape, dtype=np.uint8)
    status[labels > 0] = 2  # Mark seed points as processed

    # Get coordinates of all labeled seed points
    labeled_coords = np.argwhere(labels > 0)
    
    # For each labeled pixel, add its unlabeled neighbors to the queue
    for x, y, z in labeled_coords:
        current_label = labels[x, y, z]
        # Check 6-connected neighbors
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < labels.shape[0] and
                0 <= ny < labels.shape[1] and
                0 <= nz < labels.shape[2]):
                if status[nx, ny, nz] == 0:
                    priority = gradient_magnitude_map[nx, ny, nz]
                    heapq.heappush(heap, (priority, nx, ny, nz, current_label))
                    status[nx, ny, nz] = 1  # Mark as in queue

    # Process the priority queue
    while heap:
        priority, x, y, z, current_label = heapq.heappop(heap)
        if status[x, y, z] == 2:
            continue  # Already processed
        if labels[x, y, z] == 0:
            labels[x, y, z] = current_label
        elif labels[x, y, z] != current_label:
            # Conflict detected, mark as boundary (-1)
            labels[x, y, z] = -1
        status[x, y, z] = 2  # Mark as processed
        
        # Add neighbors to queue
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < labels.shape[0] and
                0 <= ny < labels.shape[1] and
                0 <= nz < labels.shape[2]):
                if status[nx, ny, nz] == 0:
                    heapq.heappush(heap, (gradient_magnitude_map[nx, ny, nz], nx, ny, nz, labels[x, y, z]))
                    status[nx, ny, nz] = 1  # Mark as in queue

    return labels



from gradient_magnitude_map import compute_gradient_map_gaussian
from watershed_by_flooding import watershed_by_flooding


def compute_and_save_singlesub(subject_id: str,
                                path_fmri: str,
                                path_roi: str,
                                path_mask: str,
                                outdir_grad_map: str,
                                outdir_sim_mtrx: str,
                                outdir_parcel: str):
    # extract data and affine transformation matrix
    fmri_data, _ = extract_4Ddata_from_nii(path_fmri)
    roi_data, original_affine = extract_3Ddata_from_nii(path_roi)
    mask_data, _ = extract_3Ddata_from_nii(path_mask)
    
    # Check the dimensions of the data
    if not fmri_data.shape[:-1] == mask_data.shape == roi_data.shape:
        print('fmri shape,' , fmri_data.shape[:-1])
        print('mask shape,' , mask_data.shape)
        print('roi shape,' , roi_data.shape)
        sys.exit('Data dimensions do not match. Exiting.')
    # Preprocess the ROI data
    roi_data = preprocess_ROI_data(roi_data, mask_data, fmri_data)
    nVoxels = np.prod(roi_data.shape)

    # 1 :: Compute the similarity matrix
    sim_matrix, spatial_position = compute_similarity_matrix_in_ROI(fmri_data, roi_data, mask_data)
    # Save the similarity matrix
    out_base_name = f'similarity_matrix_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    # Ensure the output directory exists
    os.makedirs(outdir_sim_mtrx, exist_ok=True)
    # Prepare the dictionary to save both sim_matrix and spatial_position
    data_to_save = {
        'S': sim_matrix,
        'spatial_position': spatial_position
    }
    scipy.io.savemat(os.path.join(outdir_sim_mtrx, out_base_name), data_to_save)
    print('Similarity matrix saved.')
    
    
    # 2 :: Compute the gradient map
    gradient_magnitude_map = compute_gradient_map_gaussian(sim_matrix, spatial_position, roi_data.shape)
    # Save Results
    out_base_name = f'gradient_map_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    # Ensure the output directory exists
    os.makedirs(outdir_grad_map, exist_ok=True)
    # Create a NIfTI image using nibabel
    nii_img = nib.Nifti1Image(gradient_magnitude_map, affine=original_affine)
    nib.save(nii_img, os.path.join(outdir_grad_map, out_base_name + '.nii'))
    print('Gradient map saved.')
    
    
    # 4 :: Perform watershed algorithm
    labels = watershed_by_flooding(gradient_magnitude_map, seeds_maps)
    # Save the parcellation map
    out_base_name = f'parcellation_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    # Ensure the output directory exists
    os.makedirs(outdir_parcel, exist_ok=True)
    nii_img = nib.Nifti1Image(labels, affine=original_affine)
    nib.save(nii_img, os.path.join(outdir_parcel, out_base_name + '.nii'))
    print('Parcellation map saved.')
    print('Done.')
    
    return None



if __name__ == '__main__':
    """The path are hard coded for now, but they can be changed to be given as input arguments.
    """
    subject_id = 'S01'
    # Input files path
    path_fmri = "G:/HCP/func/rfMRI_REST1_LR.nii.gz"
    path_roi = 'G:/RSFC/ROI_data/rROI_postcentral.nii'
    path_mask = 'G:/RSFC/ROI_data/rMASK_wholebrain.nii'
    
    # Output file path
    outdir_grad_map ='G:/HCP/outputs/grad_maps'
    outdir_sim_mtrx = 'G:/HCP/outputs/sim_mtrx'
    outdir_parcel = 'G:/HCP/outputs/parcels'

    # Extract data and affine transformation matrix
    compute_and_save_singlesub(subject_id,
                                path_fmri,
                                path_roi,
                                path_mask,
                                outdir_grad_map,
                                outdir_sim_mtrx,
                                outdir_parcel)

#%%
if __name__ == '__main__':
    # input files path
    path_fmri = 'G:/RSFC/resting_state_data/rwsraOB_TD_FBI_S01_007_Rest.nii'
    path_roi = 'G:/RSFC/ROI_data/rROI_postcentral.nii'
    path_mask = 'G:/RSFC/ROI_data/rMASK_wholebrain.nii'
    # output file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir_grad_map = os.path.join(script_dir, 'output_folder/grad_maps')
    outdir_sim_mtrx = os.path.join(script_dir, 'output_folder/sim_mtrx')
    outdir_parcel = os.path.join(script_dir, 'output_folder/parcellation_maps')
        
    # extract data and affine transformation matrix
    fmri_data, _ = extract_4Ddata_from_nii(path_fmri)
    roi_data, original_affine = extract_3Ddata_from_nii(path_roi)
    mask_data, _ = extract_3Ddata_from_nii(path_mask)
    
    # Check the dimensions of the data
    if not fmri_data.shape[:-1] == mask_data.shape == roi_data.shape:
        sys.exit('Data dimensions do not match. Exiting.')
    # Preprocess the ROI data
    roi_data = preprocess_ROI_data(roi_data, mask_data, fmri_data)    
    nVoxels = np.prod(roi_data.shape)
    
    # 1 :: Compute the similarity matrix
    load_sim_matrix = True
    if load_sim_matrix:
        print('Loading similarity matrix...')
        # load the similarity matrix
        file_path_S = 'G:/RSFC/output_folder/similarity_matrix_S02.eta2'
        sim_matrix = scipy.io.loadmat(file_path_S)['S']
        spatial_position = scipy.io.loadmat(file_path_S)['spatial_position']
    else:
        sim_matrix, spatial_position = compute_similarity_matrix_in_ROI(fmri_data, roi_data, mask_data)
        # Save the similarity matrix
        out_base_name = f'similarity_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
        # Ensure the output directory exists
        os.makedirs(outdir_sim_mtrx, exist_ok=True)
        output_path = os.path.join(outdir_sim_mtrx, out_base_name)
        # Prepare the dictionary to save both sim_matrix and spatial_position
        data_to_save = {
            'S': sim_matrix,
            'spatial_position': spatial_position
        }
        # Save the data using savemat
        scipy.io.savemat(output_path, data_to_save)
        print('Similarity matrix saved.')
    
    
    #%% 2 :: Compute the gradient map
    # laod the gradient map
    load_grad_map = True
    if load_grad_map:
        print('Loading gradient map...')
        # load the gradient map
        file_path_G = 'C:/Users/Robin/Documents/1_EPFL/PDMe/repo/brain_parcellation/output_folder/grad_maps/gradient_map_20241011_102827.nii'
        gradient_magnitude_map = nib.load(file_path_G).get_fdata()
    else:
        gradient_magnitude_map = compute_gradient_map(sim_matrix, spatial_position, roi_data.shape)
        # Save Results
        save_grad_map = False
        if save_grad_map:
            out_base_name = f'gradient_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
            # Ensure the output directory exists
            os.makedirs(outdir_grad_map, exist_ok=True)
            output_path = os.path.join(outdir_grad_map, out_base_name + '.nii')
            # Create a NIfTI image using nibabel
            nii_img = nib.Nifti1Image(gradient_magnitude_map, affine=original_affine)
            # Save the NIfTI image to the output path
            nib.save(nii_img, output_path)
    
    
    #%% 3 :: Find the seeds
    seeds_maps = find_seeds(gradient_magnitude_map)
    
    #%% 4 :: Perform watershed algorithm
    labels = watershed_by_flooding(gradient_magnitude_map, seeds_maps)
    
    # TODO  USES THIS CODE FOR THE FINAL preprocess the map for watershed by flooding
    # indices = np.where(gradient_magnitude_map==0)
    # gradient_magnitude_map2 = gradient_magnitude_map
    # gradient_magnitude_map2[indices] = np.inf
    # labels = watershed_by_flooding(gradient_magnitude_map2, seeds_maps)

    #save the parcellation map
    out_base_name = f'parcellation_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    os.makedirs(outdir_parcel, exist_ok=True)
    output_path = os.path.join(outdir_parcel, out_base_name + '.nii')
    nii_img = nib.Nifti1Image(labels, affine=original_affine)
    nib.save(nii_img, output_path)
    print('Parcellation map saved.')
    
    
    
    #%% Show the seeds postiion and their value 
    seed_pos = np.where(seeds_maps > 0)
    print('Seeds position and their value:')
    magnitude = gradient_magnitude_map[seed_pos]
    
# %%


#TODO list
# - perform ICA-FIX on each subject
# - Implement the group gradient map

# - Implement the watershed algorithm
# - Implement the watershed algorithm using flooding




# %% trash can 
"""
def watershed_by_flooding(gradient_magnitude_map: np.ndarray, 
                          seeds_maps: np.ndarray) -> np.ndarray:

    import heapq
    from scipy import ndimage
    print('Performing watershed by flooding...')
    # Initialize labels array
    
    # Label connected components in seeds_maps
    labels, num_labels = ndimage.label(seeds_maps > 0)
    
    # Initialize priority queue
    # Elements are tuples: (priority, x, y, z, label)
    heap = []
    
    # Create a status array to keep track of pixels
    # status == 0: unprocessed, status == 1: in queue, status == 2: processed
    status = np.zeros(labels.shape, dtype=np.uint8)
    status[labels > 0] = 2  # Mark seed points as processed

    # Get coordinates of all labeled seed points
    labeled_coords = np.argwhere(labels > 0)
    
    # For each labeled pixel, add its unlabeled neighbors to the queue
    for x, y, z in labeled_coords:
        current_label = labels[x, y, z]
        # Check 6-connected neighbors
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < labels.shape[0] and
                0 <= ny < labels.shape[1] and
                0 <= nz < labels.shape[2]):
                if status[nx, ny, nz] == 0:
                    priority = gradient_magnitude_map[nx, ny, nz]
                    heapq.heappush(heap, (priority, nx, ny, nz, current_label))
                    status[nx, ny, nz] = 1  # Mark as in queue

    # Process the priority queue
    while heap:
        priority, x, y, z, current_label = heapq.heappop(heap)
        if status[x, y, z] == 2:
            continue  # Already processed
        if labels[x, y, z] == 0:
            labels[x, y, z] = current_label
        elif labels[x, y, z] != current_label:
            # Conflict detected, mark as boundary (-1)
            labels[x, y, z] = -1
        status[x, y, z] = 2  # Mark as processed
        
        # Add neighbors to queue
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < labels.shape[0] and
                0 <= ny < labels.shape[1] and
                0 <= nz < labels.shape[2]):
                if status[nx, ny, nz] == 0:
                    heapq.heappush(heap, (gradient_magnitude_map[nx, ny, nz], nx, ny, nz, labels[x, y, z]))
                    status[nx, ny, nz] = 1  # Mark as in queue

    return labels

"""