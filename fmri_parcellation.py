
#%%

import sys
import os
from datetime import datetime
import numpy as np
import nibabel as nib
import scipy.io
import matplotlib.pyplot as plt

from toolbox_parcellation import extract_4Ddata_from_nii, extract_3Ddata_from_nii

from gradient_magnitude_map import compute_gradient_map_gaussian

from similarity_matrix import fingerprint_simmatrix_in_ROI, \
                              simple_simmatrix_in_ROI
                                        
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

def compute_and_save_gradient_map_single_subject(subject_id: str,
                                                path_fmri: str,
                                                path_roi: str,
                                                path_mask: str,
                                                outdir_grad_map: str,
                                                outdir_sim_mtrx: str,
                                                outdir_parcel: str):
    
    pass

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
    sim_matrix, spatial_position = simple_simmatrix_in_ROI(fmri_data, roi_data)
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
    labels = watershed_by_flooding(gradient_magnitude_map)
    # Save the parcellation map
    out_base_name = f'parcellation_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    # Ensure the output directory exists
    os.makedirs(outdir_parcel, exist_ok=True)
    nii_img = nib.Nifti1Image(labels, affine=original_affine)
    nib.save(nii_img, os.path.join(outdir_parcel, out_base_name + '.nii'))
    print('Parcellation map saved.')
    print('Done.')
    
    # TODO :OPTINAL : create a boundary map
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
    subject_id = 'S02'
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



#TODO list
# - perform ICA-FIX on each subject
# - Implement the group gradient map

# - Implement the watershed algorithm
# - Implement the watershed algorithm using flooding


# %%
