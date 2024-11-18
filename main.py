
#%%

import sys
import os
from datetime import datetime
import numpy as np
import nibabel as nib
import scipy.io
import matplotlib.pyplot as plt

from toolbox_parcellation import extract_4Ddata_from_nii, extract_3Ddata_from_nii, expand_mask\
                                ,extract_nii_files

from gradient_magnitude_map import custom_gradient_map_gaussian, \
                                    pipeline_wig2014

from similarity_matrix import fingerprint_simmatrix_in_ROI, \
                              simple_simmatrix_in_ROI
                        
from watershed_by_flooding import watershed_by_flooding, find_seeds





def preprocess_ROI_data(roi_mask: np.ndarray,
                        brain_mask: np.ndarray,
                        fmri_data: np.ndarray) -> np.ndarray:
    """Preprocess the ROI data"""
    # Keep only the ROI voxels inside the mask
    roi_mask = roi_mask * brain_mask
    # Keep only the ROI voxels that have positive variace in the fMRI data
    voxel_variances = np.var(fmri_data, axis=3)
    roi_mask = roi_mask * (voxel_variances > 0)
    return roi_mask



def compute_and_save_singlesub(subject_id: str,
                                fmri_path: str,
                                roi_mask_path: str,
                                brain_mask_path: str,
                                outdir_grad_map: str,
                                outdir_sim_mtrx: str,
                                outdir_parcel: str):
    # extract data and affine transformation matrix
    fmri_data, roi_mask, brain_mask, original_affine = extract_nii_files(fmri_path, roi_mask_path, brain_mask_path, output_dir)
    # Expand the mask size to avoid border issues with the gradient map
    extended_roi_mask = expand_mask(roi_mask, expansion_voxels=3)
    
    # Check the dimensions of the data
    if not fmri_data.shape[:-1] == brain_mask.shape == extended_roi_mask.shape:
        print('fmri shape,' , fmri_data.shape[:-1])
        print('mask shape,' , brain_mask.shape)
        print('roi shape,' , roi_mask.shape)
        sys.exit('Data dimensions do not match. Exiting.')
    # Preprocess the ROI data
    print('Preprocessing the ROI data...')
    extended_roi_mask = preprocess_ROI_data(extended_roi_mask, brain_mask, fmri_data)
    roi_mask = preprocess_ROI_data(roi_mask, brain_mask, fmri_data)
    print('ROI data preprocessed.')
    nVoxels = np.prod(extended_roi_mask.shape)

    # 1 :: Compute the similarity matrix
    print('Start similarity matrix computation...')
    sim_matrix, spatial_position = fingerprint_simmatrix_in_ROI(fmri_data, 
                                                                extended_roi_mask,
                                                                brain_mask)
    # Save the similarity matrix
    out_base_name = f'similarity_matrix_{subject_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
    print('Start gradient map computation...')
    mean_edge_map = pipeline_wig2014(sim_matrix,
                                    spatial_position,
                                    extended_roi_mask.shape)
    # Save Results
    out_base_name = f'mean_edge_map_{subject_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # Ensure the output directory exists
    os.makedirs(outdir_grad_map, exist_ok=True)
    # Create a NIfTI image using nibabel
    nii_img = nib.Nifti1Image(mean_edge_map, affine=original_affine)
    nib.save(nii_img, os.path.join(outdir_grad_map, out_base_name + '.nii'))
    print('Gradient map saved.')
    # TODO code for loading Specify the path to the NIfTI file you want to load
    # file_path = os.path.join(outdir_grad_map, out_base_name + '.nii')
    # nii_img = nib.load(file_path)
    # mean_edge_map = nii_img.get_fdata()
    # original_affine = nii_img.affine
    
    # 3.0 :: restric the edge map to the ROI (otherwise it will be the extended ROI)
    seeds = find_seeds(mean_edge_map,
                       roi_mask)
    # 3 :: Perform watershed algorithm
    labels = watershed_by_flooding(mean_edge_map,
                                   seeds,
                                   roi_mask,
                                   flooding_percent=100)
    # Save the parcellation map
    out_base_name = f'parcellation_map_{subject_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # Ensure the output directory exists
    os.makedirs(outdir_parcel, exist_ok=True)
    nii_img = nib.Nifti1Image(labels, affine=original_affine)
    nib.save(nii_img, os.path.join(outdir_parcel, out_base_name + '.nii'))
    print('Parcellation map saved.')
    print('Done.')
    # TODO code for loading Specify the path to the NIfTI file you want to load
    # file_path = os.path.join(outdir_parcel, '...nii')
    # nii_img = nib.load(file_path)
    # parcellation_map = nii_img.get_fdata()
    # original_affine = nii_img.affine
    
    
    
    # TODO :OPTINAL : create a boundary map
    # from gradient_magnitude_map import create_boundary_maps
    # boundary_map = create_boundary_maps(labels)
    # out_base_name = f'boundary_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    # os.makedirs(outdir_parcel, exist_ok=True)
    # nii_img = nib.Nifti1Image(boundary_map, affine=original_affine)
    # nib.save(nii_img, os.path.join(outdir_parcel, out_base_name + '.nii'))
    # print('boundary map saved.')
    return None

def main(fmri_file, 
         roi_file, 
         mask_file,
         subject_id, 
         output_dir):
    
    # Output file path
    outdir_grad_map = output_dir + r'/outputs/edge_map'
    outdir_sim_mtrx = output_dir + r'/outputs/sim_mtrx'
    outdir_parcel =   output_dir + r'/outputs/parcels'
    compute_and_save_singlesub(subject_id,
                                fmri_file,
                                roi_file,
                                mask_file,
                                outdir_grad_map,
                                outdir_sim_mtrx,
                                outdir_parcel)
    
    return None

#%%
if __name__ == '__main__':
    """The path are hard coded for now, but they can be changed to be given as input arguments.
    """
    for i in range(10,21):
        # Input files path
        if i == 1:
            fmri_path = f'G:/DATA_min_preproc/dataset_study2/sub-{i:02d}/func/rwsraOB_TD_FBI_S{i:02d}_007_Rest.nii'
        else:
            fmri_path = f'G:/DATA_min_preproc/dataset_study2/sub-{i:02d}/func/rwsraOB_TD_FBI_S{i:02d}_006_Rest.nii' # fMRI data
        roi_mask_path = f'G:/MASK_standard/gm_postcentral_mask.nii' # Extraction of the grey matter in S1
        brain_mask_path = f'G:/MASK_standard/MNI152_T1_2mm_brain_mask.nii' # Whole brain mask MNI152 
        # Output file path
        output_dir = f'G:/DATA_min_preproc/dataset_study2/sub-{i:02d}/'

        # Extract data and affine transformation matrix
        main(fmri_path,
            roi_mask_path,
            brain_mask_path,
            f'S{i}',
            output_dir)

# %%
outdir_parcel = 'G:/DATA_min_preproc/dataset_study1/S04/outputs/parcels'
file_path = os.path.join(outdir_parcel, 'parcellation_map_S04_20241112_210105.nii')
nii_img = nib.load(file_path)
parcellation_map = nii_img.get_fdata()
original_affine = nii_img.affine