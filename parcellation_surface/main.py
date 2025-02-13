#TODO : this file is for the entire pipeline computation
#%%
import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

from preprocessing_surface import load_volume_data, fmri_vol2surf
from similarity_matrix import compute_similarity_matrix
from smoothing import smooth_surface, smooth_surface_graph
from gradient import compute_gradients, compute_gradient_average, build_mesh_graph, \
                     save_gradient_mgh, load_gradient_mgh
from watershed import watershed_by_flooding, save_labels_mgh \
                    , load_labels_mgh
from visualization import visualize_brain_surface




#%%

    
#####################################################################################################################



def full_pipeline():
    # Global configuration defined once
    config = {
        "fsavg6_dir": Path(r"D:\Data_Conn_Preproc\PPSFACE_N18\fsaverage6"),
        "subjects_dir": Path(r"D:\Data_Conn_Preproc\PPSFACE_N20")
    }
    
    for subject_num in range(1, 21):
        subject = f"{subject_num:02d}"
        print(f"Processing subject {subject}")
        
        subj_dir = config["subjects_dir"] / f"sub-{subject}"
        # Create output directory if it doesn't exist.
        (subj_dir / "outputs_surface").mkdir(exist_ok=True, parents=True)
        
        for hemisphere in ['lh', 'rh']:
            # Define paths using pathlib
            surf_fmri_path = subj_dir / "func" / f"surf_conn_sub{subject}_run2_{hemisphere}.func.fsaverage6.mgh"
            surface_path = config["fsavg6_dir"] / "surf" / f"{hemisphere}.white"
            vol_fmri_file = subj_dir / "func" / f"conn_wsraPPSFACE_sub-{subject}_run2.nii.gz"
            brain_mask_path = subj_dir / f"sub{subject}_freesurfer" / "mri" / "brainmask.mgz"
            
            # LOAD SURFACE GEOMETRY
            coords, faces = nib.freesurfer.read_geometry(str(surface_path))
            
            # LOAD SURFACE DATA
            surf_fmri_img = nib.load(str(surf_fmri_path))
            surf_fmri = np.squeeze(surf_fmri_img.get_fdata()).astype(np.float32)
            # Normalize the data (mandatory)
            surf_fmri = (surf_fmri - np.mean(surf_fmri, axis=1, keepdims=True)) / np.std(surf_fmri, axis=1, keepdims=True)
            print(f"Surf data shape (2D): {surf_fmri.shape}")
            
            # Smooth the surf_fmri
            surf_fmri = smooth_surface(faces, surf_fmri, iterations=5)
            
            # LOAD VOLUME DATA
            vol_fmri, resampled_mask, affine = load_volume_data(str(vol_fmri_file),
                                                                str(brain_mask_path))
            print(f"Volume data shape: {vol_fmri.shape}")
            
            # BUILD MESH GRAPH
            graph = build_mesh_graph(faces)
            print(f"Number of vertices: {coords.shape[0]}")
            print(f"Number of faces: {faces.shape[0]}")
            
            # COMPUTE SIMILARITY MATRIX
            similarity_matrix = compute_similarity_matrix(surf_fmri, vol_fmri, resampled_mask, n_modes=380)
            del surf_fmri, vol_fmri  # Save memory
            
            print('Smoothing similarity matrix...')
            sim_matrix_smoothed = smooth_surface_graph(graph, similarity_matrix, iterations=10)
            del similarity_matrix  # Save memory
            
            print('Computing gradients...')
            gradients = compute_gradients(graph, sim_matrix_smoothed)
            gradients_sum = gradients.sum(axis=1)
            
            # SAVE GRADIENT MAP
            print('Saving gradients...')
            output_grad_path = subj_dir / "outputs_surface" / f"gradient_maprun2_{hemisphere}.mgh"
            save_gradient_mgh(gradients_sum, str(output_grad_path), hemisphere=hemisphere, name="grad_conn_fsavg6")
            
            # Further smoothing of gradient if needed
            gradient_smoothed = smooth_surface(faces, gradients_sum, iterations=10)
            
            # COMPUTE THE EDGE MAP (e.g., watershed segmentation)
            labels = watershed_by_flooding(graph, gradient_smoothed)
            
            # SAVE LABELS
            print('Saving labels...')
            output_label_path = subj_dir / "outputs_surface" / f"labelsrun2_{hemisphere}.mgh"
            save_labels_mgh(labels, str(output_label_path), hemisphere=hemisphere, name="labels_conn_fsavg6")
            
            print(f"Subject {subject}, hemisphere {hemisphere} finished")
#%%
# def load_config(config_path):
#     """Load YAML config file"""
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)

#%%

if __name__ == "__main__":
    print("Load and Test the results")
    
    # Path for the dataset 2
    # path_midthickness = "D:\DATA_min_preproc\dataset_study2\sub-04\sub04_freesurfer\surf\lh.midthickness.32k.surf.gii"
    # path_midthickness_inflated = "D:\DATA_min_preproc\dataset_study2\sub-04\sub04_freesurfer\surf\lh.midthickness.inflated.32k.surf.gii"
    # path_gradient_values = "D:\DATA_min_preproc\dataset_study2\sub-04\outputs_surface\gradient_map\lh_gradient_map_20250106104701.npy"
    # path_parcels_values = "D:\DATA_min_preproc\dataset_study2\sub-04\outputs_surface\labels\lh_labels_20250106104701.npy"
    # # Path for the dataset 1
    # path_midthickness = "D:\DATA_min_preproc\dataset_study1\sub-03\sub03_freesurfer\surf\lh.midthickness.32k.surf.gii"
    # path_midthickness_inflated = "D:\DATA_min_preproc\dataset_study1\sub-03\sub03_freesurfer\surf\lh.midthickness.inflated.32k.surf.gii"
    # path_gradient_values = "D:\DATA_min_preproc\dataset_study1\sub-03\outputs_surface\gradient_map\lh_gradient_map_20250106155120.npy"
    # path_parcels_values = "D:\DATA_min_preproc\dataset_study1\sub-03\outputs_surface\labels\lh_labels_20250106155121.npy"
    
    # surface_mesh_path = r"D:\Data_Conn_Preproc\PPSFACE_N18\fsaverage6\surf\lh.inflated"
    # gradient_mgh_path = r"D:\Data_Conn_Preproc\PPSFACE_N18\sub-01\outputs_surface\gradient_map\grad_conn_fsavg6_lh_20250130105744.mgh"
    # labels_mgh_path = r"D:\Data_Conn_Preproc\PPSFACE_N18\sub-01\outputs_surface\labels\labels_conn_fsavg6_lh_20250130105745.mgh"
    # coords, faces = nib.freesurfer.read_geometry(surface_path)
    # gradient_values = load_gradient_mgh(path_gradient_values)
    # parcels_values = load_labels_mgh(path_parcels_values)
    # # visualize_brain_surface(coords, faces, gradient_values, title='Gradient Map')
    # visualize_brain_surface(coords, faces, parcels_values, title='Parcellation Map')
    
    path_inflated = r"D:\Data_Conn_Preproc\PPSFACE_N18\fsaverage6\surf\lh.inflated"
    coords_, faces_ = nib.freesurfer.read_geometry(path_inflated)
    gradient_path = r"D:\Data_Conn_Preproc\PPSFACE_N18\sub-10\outputs_surface\gradient_maprun2\grad_conn_fsavg6_lh_20250201103221.mgh"
    gradient = load_gradient_mgh(gradient_path)
    visualize_brain_surface(coords_, faces_, gradient)
    
    
    parser = argparse.ArgumentParser(description="Run parcellation pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--subjects", nargs="+", required=True, help="List of subjects to process.")
    
    args = parser.parse_args()
    config = load_config(args.config)
    for subject in args.subjects:
        single_subj_parcellation_fsaverage(subject, config)
        
    # TODO : create a config file, each subject must have 3 folders: func, anat, sub{01}freesurfer
    # python run_pipeline.py --config config.yaml --subjects 01 02 03 04 05
    
#%% Paths (TODO : REMOVE THESE GLOBAL VARIABLES)
# subject = r"04"
# subj_dir = r"D:\DATA_min_preproc\dataset_study2\sub-" + subject

# path_func = subj_dir + r"\func\rwsraOB_TD_FBI_S" + subject + r"_006_Rest.nii"
# path_midthickness_r = subj_dir + r"\sub" + subject + r"_freesurfer\surf\rh.midthickness.32k.surf.gii"
# path_midthickness_l = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.midthickness.32k.surf.gii"

# path_white_r = subj_dir + r"\sub" + subject + r"_freesurfer\surf\rh.white.32k.surf.gii"
# path_white_l = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.white.32k.surf.gii"

# path_pial_r = subj_dir + r"\sub" + subject + r"_freesurfer\surf\rh.pial.32k.surf.gii"
# path_pial_l = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.pial.32k.surf.gii"

# path_midthickness_l_inflated = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.midthickness.inflated.32k.surf.gii"
# path_midthickness_r_inflated = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.midthickness.inflated.32k.surf.gii"

# path_brain_mask = subj_dir + r"\sub" + subject + r"_freesurfer\mri\brainmask.mgz"    
    
    
#     def single_subj_parcellation_native(subj_dir, 
#                              path_func, 
#                              path_midthickness_l, 
#                              path_midthickness_r,
#                              path_brain_mask):
#     """Performa  parcellation on a single subject in native space. Within their personal midthickness surface.
#     Args:
#         subj_dir (_type_): _description_
#         path_func (_type_): _description_
#         path_midthickness_l (_type_): _description_
#         path_midthickness_r (_type_): _description_
#         path_brain_mask (_type_): _description_
#     Raises:
#         ValueError: _description_
#     """
#     print(f"Processing subject {subj_dir}")
    
#     for hemisphere in ['lh', 'rh']:
#         print(f'{hemisphere} loading data...')
#         vol_fmri, resampled_mask_img, affine = load_volume_data(path_func,
#                                                                 path_brain_mask)
        
#         surf_fmri_l, surf_fmri_r = fmri_vol2surf(nib.load(path_func), 
#                                                 path_midthickness_l, 
#                                                 path_midthickness_r)
#         if hemisphere == 'lh':
#             surf_fmri = surf_fmri_l
#             path_midthickness = path_midthickness_l
#             # path_midthickness_inflated = path_midthickness_l_inflated
#         elif hemisphere == 'rh':    
#             surf_fmri = surf_fmri_r
#             path_midthickness = path_midthickness_r
#             # path_midthickness_inflated = path_midthickness_r_inflated
#         else:
#             raise ValueError('Invalid hemisphere') # Debug inside the function
#         del surf_fmri_l, surf_fmri_r # Save memory
        
#         gii = nib.load(path_midthickness)
#         coords = gii.darrays[0].data  # shape: (N_vertices, 3)
#         faces = gii.darrays[1].data   # shape: (N_faces, 3)
#         graph = build_mesh_graph(faces)

#         print(f'{hemisphere} computing similarity matrix...')
#         similarity_matrix = compute_similarity_matrix(surf_fmri, 
#                                                     vol_fmri,
#                                                     resampled_mask_img,
#                                                     n_modes=179)
#         print(f'{hemisphere} smooth similarty matrix...')
#         sim_matrix_smooothed = smooth_surface(faces,
#                                             similarity_matrix, 
#                                             iterations=5)
#         del similarity_matrix # Save memory
#         print(f'{hemisphere} computing gradients...')
#         gradients = compute_gradients(graph,
#                                     sim_matrix_smooothed,
#                                     skip=20)
        
#         save_gradient_mgh(gradients,
#                         subj_dir + r"\outputs_surface\gradient_map",
#                         hemisphere=hemisphere)
#         print(f'{hemisphere} gradients saved...')
#         gradient_smoothed = smooth_surface(faces,
#                                         gradients,
#                                         iterations=5)
        
#         print(f'{hemisphere} computing parcellation map...')
#         # Compute the edge map
#         labels = watershed_by_flooding(graph, gradient_smoothed)
#         # Saving the labels
#         save_labels_mgh(labels,
#                         subj_dir + r"\outputs_surface\labels",
#                         hemisphere =hemisphere)
#         print(f'{hemisphere} parcellation map saved...')
#         visualize_brain_surface(coords, faces, labels)


# def single_subj_parcellation_fsaverage(subject, config):
#     """
#     Need the following files: 
#     -fmri data stored in a nii.gz file
#     -left/right projection of fmri data into fsaverage6 space.
#     -the fsaverage6 folder from freesurfer
    
#     Returns:
#         _type_: _description_
#     """
#     print(f"Processing subject {subject}")
#     #%%
#     subject = f"01"
#     config = {"fsavg6_dir": r"D:\Data_Conn_Preproc\PPSFACE_N18\fsaverage6",
#               "subjects_dir": r"D:\Data_Conn_Preproc\PPSFACE_N18",} # remove
#     fsavg6_dir = config["fsavg6_dir"]
#     subjects_dir = config["subjects_dir"]
    
#     hemisphere = 'lh' # remove
#     #%%
#     for hemisphere in ['lh', 'rh']:
#         #%%
#         subj_dir = subjects_dir + r"\sub-" + subject
#         surf_fmri_path = subj_dir + r"\func\surf_conn_sub" + subject + f"_run1_{hemisphere}.func.fsaverage6.mgh"
#         surf_fmri_path = subj_dir + r"\func\surf_conn_sub" + subject + f"_run1_{hemisphere}.func.fsaverage6.mgh"
#         # TODO : WARNING CUSTOMIZE THE PATH BASED ON YOUR DATA
#         surface_path = fsavg6_dir + f"\\surf\\{hemisphere}.white" # Path of surface geometry
#         vol_fmri_file = subj_dir + r"\func\conn_wsraPPSFACE_sub-" + subject + r"_run1.nii.gz"
#         brain_mask_path = subj_dir + r"\sub" + subject + r"_freesurfer\mri\brainmask.mgz"
        
#         # LOAD SURFACE GEOMETRY
#         coords, faces = nib.freesurfer.read_geometry(surface_path)
#         # LOAD SURFACE DATA
#         surf_fmri_img = nib.load(surf_fmri_path)
#         surf_fmri = surf_fmri_img.get_fdata()
#         surf_fmri = np.squeeze(surf_fmri)
#         surf_fmri = (surf_fmri - np.mean(surf_fmri, axis=1, keepdims=True)) \
#                     / np.std(surf_fmri, axis=1, keepdims=True) # Normalize the data (MENDATORY)
#         print(f"Surf data shape (2D): {surf_fmri.shape}")
        
#         # smooth the surf_fmri
#         surf_fmri = smooth_surface(faces, surf_fmri, iterations=5)
        
#         # LOAD VOLUME DATA
#         vol_fmri, resampled_mask, affine = load_volume_data(vol_fmri_file,
#                                                             brain_mask_path)
#         print(f"Volume data shape: {vol_fmri.shape}")
#         # LOAD FS6 DATA
#         graph = build_mesh_graph(faces)
#         print(f"Number of vertices: {coords.shape[0]}")
#         print(f"Number of faces: {faces.shape[0]}")
        
#         # Normalized the fmri data and extract the spatial modes
#         similarity_matrix = compute_similarity_matrix(surf_fmri, 
#                                                     vol_fmri,
#                                                     resampled_mask,
#                                                     n_modes=380)
#         #
#         # visualize_brain_surface(coords, faces, similarity_matrix[0,:])
#         # Smooth the similarity matrix
#         print(f'smooth similarty matrix...')
#         sim_matrix_smooothed = smooth_surface(faces,
#                                               similarity_matrix, 
#                                               iterations=10)
#         #visualize_brain_surface(coords, faces, sim_matrix_smooothed[:,0])
#         del similarity_matrix # Save memory
        
#         print(f'computing gradients...')
#         gradients = compute_gradient_average(graph,
#                                     sim_matrix_smooothed,
#                                     skip=40)
#         # save the gradient map
#         print(f'saving gradients...')
#         save_gradient_mgh(gradients,
#                         subj_dir + r"\outputs_surface\gradient_map",
#                         hemisphere=hemisphere,
#                         name=f"grad_conn_fsavg6")
        
#         gradient_smoothed = smooth_surface(faces,
#                                             gradients,
#                                             iterations=10)
        
#         # Compute the edge map
#         labels = watershed_by_flooding(graph, gradient_smoothed)
#         # Saving the labels
#         print(f'saving labels...')
#         save_labels_mgh(labels,
#                         subj_dir + r"\outputs_surface\labels",
#                         hemisphere=hemisphere,
#                         name=f"labels_conn_fsavg6")
#         print(f'sub-{subject} hemis : {hemisphere} finished')
#     # visualize_brain_surface(coords, faces, labels)


# def new_pipeline():
#     print(f"Processing subject {subject}")
#     #%%
#     for subject in range(1, 11):
#         subject = f"{subject:02d}"
#         config = {"fsavg6_dir": r"D:\Data_Conn_Preproc\PPSFACE_N18\fsaverage6",
#                   "subjects_dir": r"D:\Data_Conn_Preproc\PPSFACE_N18",}
#         fsavg6_dir = config["fsavg6_dir"]
#         subjects_dir = config["subjects_dir"]
#         #           
#         # subject = f"01"
#         # config = {"fsavg6_dir": r"D:\Data_Conn_Preproc\PPSFACE_N18\fsaverage6",
#         #         "subjects_dir": r"D:\Data_Conn_Preproc\PPSFACE_N18",} # remove
#         # fsavg6_dir = config["fsavg6_dir"]
#         # subjects_dir = config["subjects_dir"]
        
#         # hemisphere = 'lh' # remove
#         #
#         for hemisphere in ['lh', 'rh']:
#             #
#             subj_dir = subjects_dir + r"\sub-" + subject
#             surf_fmri_path = subj_dir + r"\func\surf_conn_sub" + subject + f"_run2_{hemisphere}.func.fsaverage6.mgh"
#             surf_fmri_path = subj_dir + r"\func\surf_conn_sub" + subject + f"_run2_{hemisphere}.func.fsaverage6.mgh"
#             # TODO : WARNING CUSTOMIZE THE PATH BASED ON YOUR DATA
#             surface_path = fsavg6_dir + f"\\surf\\{hemisphere}.white" # Path of surface geometry
#             vol_fmri_file = subj_dir + r"\func\conn_wsraPPSFACE_sub-" + subject + r"_run2.nii.gz"
#             brain_mask_path = subj_dir + r"\sub" + subject + r"_freesurfer\mri\brainmask.mgz"
            
#             # LOAD SURFACE GEOMETRY
#             coords, faces = nib.freesurfer.read_geometry(surface_path)
#             # LOAD SURFACE DATA
#             surf_fmri_img = nib.load(surf_fmri_path)
#             surf_fmri = surf_fmri_img.get_fdata()
#             surf_fmri = np.squeeze(surf_fmri)
#             surf_fmri = (surf_fmri - np.mean(surf_fmri, axis=1, keepdims=True)) \
#                         / np.std(surf_fmri, axis=1, keepdims=True) # Normalize the data (MENDATORY)
#             print(f"Surf data shape (2D): {surf_fmri.shape}")
            
#             # smooth the surf_fmri
#             surf_fmri = smooth_surface(faces, surf_fmri, iterations=5)
            
#             # LOAD VOLUME DATA
#             vol_fmri, resampled_mask, affine = load_volume_data(vol_fmri_file,
#                                                                 brain_mask_path)
#             print(f"Volume data shape: {vol_fmri.shape}")
#             # LOAD FS6 DATA
#             graph = build_mesh_graph(faces)
#             print(f"Number of vertices: {coords.shape[0]}")
#             print(f"Number of faces: {faces.shape[0]}")
            
#             # Normalized the fmri data and extract the spatial modes
#             # 40k x 40k similarity matrix
#             similarity_matrix = compute_similarity_matrix(surf_fmri, 
#                                                         vol_fmri,
#                                                         resampled_mask,
#                                                         n_modes=380)
#             del surf_fmri # Save memory
#             del vol_fmri # Save memory
            
            
#             # visualize_brain_surface(coords, faces, similarity_matrix[:,0])
#             # Smooth the similarity matrix
#             print(f'smooth similarty matrix...')
#             sim_matrix_smooothed = smooth_surface_with_graph_adjlist(graph,
#                                                                 similarity_matrix, 
#                                                                 iterations=10)
#             #visualize_brain_surface(coords, faces, sim_matrix_smooothed[:,0])
#             del similarity_matrix # Save memory
            
#             print(f'computing gradients...')
#             # 40k gradient maps
#             gradients = compute_gradients(graph,
#                                         sim_matrix_smooothed)
#             gradients_sum = gradients.sum(axis=1)
#             # save the gradient map
#             print(f'saving gradients...')
#             save_gradient_mgh(gradients_sum,
#                             subj_dir + r"\outputs_surface\gradient_maprun2",
#                             hemisphere=hemisphere,
#                             name=f"grad_conn_fsavg6")
#             #        
#             gradient_smoothed = smooth_surface(faces,
#                                                 gradients_sum,
#                                                 iterations=10)
            
#             # Compute the edge map
#             labels = watershed_by_flooding(graph,
#                                         gradient_smoothed)
#             # Saving the labels
#             print(f'saving labels...')
#             save_labels_mgh(labels,
#                             subj_dir + r"\outputs_surface\labelsrun2",
#                             hemisphere=hemisphere,
#                             name=f"labels_conn_fsavg6")
#             print(f'sub-{subject} hemis : {hemisphere} finished')
            
            
