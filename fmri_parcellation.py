
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

from gradient_magnitude_map import compute_gradient_map_gaussian, compute_similarity_matrix_in_ROI
from watershed_by_flooding import watershed_by_flooding


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

def blur_gradient_map(gradient_magnitude_map: np.ndarray,
                      sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to the gradient magnitude map."""
    from scipy.ndimage import gaussian_filter
    print('Applying Gaussian blur to the gradient magnitude map...')
    blurred_map = gaussian_filter(gradient_magnitude_map, sigma=sigma)
    return blurred_map


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
    # TODO : to load Specify the path to the .mat file you want to load
    # file_path = os.path.join(outdir_sim_mtrx, out_base_name )
    # loaded_data = scipy.io.loadmat(file_path)
    # sim_matrix = loaded_data['S']
    # spatial_position = loaded_data['spatial_position']
    
    
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
    # TODO code for loading Specify the path to the NIfTI file you want to load
    # file_path = os.path.join(outdir_grad_map, out_base_name + '.nii')
    # nii_img = nib.load(file_path)
    # gradient_magnitude_map = nii_img.get_fdata()
    # original_affine = nii_img.affine
    
    # 3 :: Perform watershed algorithm
    labels = watershed_by_flooding(gradient_magnitude_map, seeds_maps)
    # Save the parcellation map
    out_base_name = f'parcellation_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    # Ensure the output directory exists
    os.makedirs(outdir_parcel, exist_ok=True)
    nii_img = nib.Nifti1Image(labels, affine=original_affine)
    nib.save(nii_img, os.path.join(outdir_parcel, out_base_name + '.nii'))
    print('Parcellation map saved.')
    print('Done.')
    
    # OPTINAL : create a boundary map
    # from gradient_magnitude_map import create_boundary_maps
    # boundary_map = create_boundary_maps(labels)
    # out_base_name = f'boundary_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    # os.makedirs(outdir_parcel, exist_ok=True)
    # nii_img = nib.Nifti1Image(boundary_map, affine=original_affine)
    # nib.save(nii_img, os.path.join(outdir_parcel, out_base_name + '.nii'))
    # print('boundary map saved.')
    return None



if __name__ == '__main__':
    """The path are hard coded for now, but they can be changed to be given as input arguments.
    """
    subject_id = 'S01'
    # Input files path
    path_fmri = "G:/HCP/func/rfMRI_REST1_LR.nii.gz"
    path_roi = 'G:/RSFC/ROI_data/ROI_postcentral.nii'
    path_mask = 'G:/RSFC/ROI_data/MASK_wholebrain.nii'
    
    # Output file path
    outdir_grad_map ='G:/HCP/outputs/grad_maps'
    outdir_sim_mtrx = 'G:/HCP/outputs/sim_mtrx'
    outdir_parcel = 'G:/HCP/outputs/parcels'

#%%
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