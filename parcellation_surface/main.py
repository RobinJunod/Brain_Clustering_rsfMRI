#TODO : this file is for the entire pipeline computation
#%%
import os
from typing import Literal
import numpy as np
import nibabel as nib

from preprocessing_surface import load_volume_data, fmri_vol2surf
from similarity_matrix import compute_similarity_matrix
from smoothing import smooth_surface
from gradient import compute_gradients, build_mesh_graph, \
                     save_gradient_mgh
from watershed import watershed_by_flooding, save_labels_mgh
from visualization import visualize_brain_surface


#%% Paths (TODO : REMOVE THESE GLOBAL VARIABLES)
subject = r"04"

subj_dir = r"D:\DATA_min_preproc\dataset_study2\sub-" + subject


path_func = subj_dir + r"\func\rwsraOB_TD_FBI_S" + subject + r"_006_Rest.nii"

path_midthickness_r = subj_dir + r"\sub" + subject + r"_freesurfer\surf\rh.midthickness.32k.surf.gii"
path_midthickness_l = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.midthickness.32k.surf.gii"

path_white_r = subj_dir + r"\sub" + subject + r"_freesurfer\surf\rh.white.32k.surf.gii"
path_white_l = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.white.32k.surf.gii"

path_pial_r = subj_dir + r"\sub" + subject + r"_freesurfer\surf\rh.pial.32k.surf.gii"
path_pial_l = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.pial.32k.surf.gii"

path_midthickness_l_inflated = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.midthickness.inflated.32k.surf.gii"
path_midthickness_r_inflated = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.midthickness.inflated.32k.surf.gii"

path_brain_mask = subj_dir + r"\sub" + subject + r"_freesurfer\mri\brainmask.mgz"

#%%
def single_subj_parcellation_native(subj_dir, 
                             path_func, 
                             path_midthickness_l, 
                             path_midthickness_r,
                             path_brain_mask):
    """Performa  parcellation on a single subject in native space. Within their personal midthickness surface.
    Args:
        subj_dir (_type_): _description_
        path_func (_type_): _description_
        path_midthickness_l (_type_): _description_
        path_midthickness_r (_type_): _description_
        path_brain_mask (_type_): _description_
    Raises:
        ValueError: _description_
    """
    print(f"Processing subject {subj_dir}")
    
    for hemisphere in ['lh', 'rh']:
        print(f'{hemisphere} loading data...')
        vol_fmri, resampled_mask_img, affine = load_volume_data(path_func,
                                                                path_brain_mask)
        
        surf_fmri_l, surf_fmri_r = fmri_vol2surf(nib.load(path_func), 
                                                path_midthickness_l, 
                                                path_midthickness_r)
        if hemisphere == 'lh':
            surf_fmri = surf_fmri_l
            path_midthickness = path_midthickness_l
            # path_midthickness_inflated = path_midthickness_l_inflated
        elif hemisphere == 'rh':    
            surf_fmri = surf_fmri_r
            path_midthickness = path_midthickness_r
            # path_midthickness_inflated = path_midthickness_r_inflated
        else:
            raise ValueError('Invalid hemisphere') # Debug inside the function
        del surf_fmri_l, surf_fmri_r # Save memory
        
        gii = nib.load(path_midthickness)
        coords = gii.darrays[0].data  # shape: (N_vertices, 3)
        faces = gii.darrays[1].data   # shape: (N_faces, 3)
        graph = build_mesh_graph(faces)

        print(f'{hemisphere} computing similarity matrix...')
        similarity_matrix = compute_similarity_matrix(surf_fmri, 
                                                    vol_fmri,
                                                    resampled_mask_img,
                                                    n_modes=179)
        print(f'{hemisphere} smooth similarty matrix...')
        sim_matrix_smooothed = smooth_surface(faces,
                                            similarity_matrix, 
                                            iterations=5)
        del similarity_matrix # Save memory
        print(f'{hemisphere} computing gradients...')
        gradients = compute_gradients(graph,
                                    sim_matrix_smooothed,
                                    skip=20)
        
        save_gradient_mgh(gradients,
                        subj_dir + r"\outputs_surface\gradient_map",
                        hemisphere=hemisphere)
        print(f'{hemisphere} gradients saved...')
        gradient_smoothed = smooth_surface(faces,
                                        gradients,
                                        iterations=5)
        
        print(f'{hemisphere} computing parcellation map...')
        # Compute the edge map
        labels = watershed_by_flooding(graph, gradient_smoothed)
        # Saving the labels
        save_labels_mgh(labels,
                        subj_dir + r"\outputs_surface\labels",
                        hemisphere =hemisphere)
        print(f'{hemisphere} parcellation map saved...')
        visualize_brain_surface(coords, faces, labels)


# single_subj_parcellation(subj_dir, 
#                         path_func, 
#                         path_midthickness_l, 
#                         path_midthickness_r,
#                         path_brain_mask)


def single_subj_parcellation_fsaverage(subj_dir, 
                                       fsavg6_dir):
    """
    Need the following files: 
    -fmri data stored in a nii.gz file
    -left/right projection of fmri data into fsaverage6 space.
    -the fsaverage6 folder from freesurfer
    
    Returns:
        _type_: _description_
    """
    #%%
    surface_path = r"D:\DATA_min_preproc\dataset_study1\fsaverage6\surf\lh.white"
    # surface_path_inflated = r"D:\DATA_min_preproc\dataset_study1\fsaverage6\surf\lh.inflated"
    # surface_path_pial = r"D:\DATA_min_preproc\dataset_study1\fsaverage6\surf\lh.pial"
    
    subject = r"01"
    hemisphere = r'rh'
    for s in range(1,19):
        subject = f"{s:02d}"
        print(f"Processing subject {subject}")
        for hemisphere in ['lh', 'rh']:
            # TODO : WARNING CUSTOMIZE THE PATH BASED ON YOUR DATA
            subj_dir = r"D:\DATA_min_preproc\dataset_study1\sub-" + subject
            vol_fmri_file = subj_dir + r"\func\wsraPPS-FACE_S" + subject + r"_005_Rest.nii"
            brain_mask_path = subj_dir + r"\sub" + subject + r"_freesurfer\mri\brainmask.mgz"
            surf_fmri_path = subj_dir + r"\func\sub" + subject + f"_{hemisphere}.func.fsaverage6.mgh"
            

            # LOAD SURFACE DATA
            surf_fmri_img = nib.load(surf_fmri_path)
            surf_fmri = surf_fmri_img.get_fdata()
            surf_fmri = np.squeeze(surf_fmri)
            surf_fmri = (surf_fmri - np.mean(surf_fmri, axis=1, keepdims=True)) \
                        / np.std(surf_fmri, axis=1, keepdims=True) # Normalize the data (MENDATORY)
            print(f"Surf data shape (2D): {surf_fmri.shape}")
            # LOAD VOLUME DATA
            vol_fmri, resampled_mask, affine = load_volume_data(vol_fmri_file,
                                                                brain_mask_path)
            print(f"Volume data shape: {vol_fmri.shape}")
            # LOAD FS6 DATA
            coords, faces = nib.freesurfer.read_geometry(surface_path)
            graph = build_mesh_graph(faces)
            print(f"Number of vertices: {coords.shape[0]}")
            print(f"Number of faces: {faces.shape[0]}")
            
            # Normalized the fmri data and extract the spatial modes
            similarity_matrix = compute_similarity_matrix(surf_fmri, 
                                                        vol_fmri,
                                                        resampled_mask,
                                                        n_modes=380)
            # visualize_brain_surface(coords, faces, similarity_matrix[0,:])
            # Smooth the similarity matrix
            print(f'smooth similarty matrix...')
            sim_matrix_smooothed = smooth_surface(faces,
                                                  similarity_matrix, 
                                                  iterations=5)
            del similarity_matrix # Save memory
            
            print(f'computing gradients...')
            gradients = compute_gradients(graph,
                                        sim_matrix_smooothed,
                                        skip=40)
            # save the gradient map
            save_gradient_mgh(gradients,
                            subj_dir + r"\outputs_surface\gradient_map_fsavg6_highsmooth",
                            hemisphere=hemisphere)
            
            gradient_smoothed = smooth_surface(faces,
                                               gradients,
                                               iterations=10)
            
            # Compute the edge map
            labels = watershed_by_flooding(graph, gradient_smoothed)
            # Saving the labels
            save_labels_mgh(labels,
                            subj_dir + r"\outputs_surface\labels_fsavg6_highsmooth",
                            hemisphere =hemisphere)
    # visualize_brain_surface(coords, faces, labels)
    
    #%%
    pass

#%% Multi-subject parcellation
def multi_subj_parcellation(dataset_dir):
    """
    This script will output all the parcellation maps for all the subjects individually.
    WARNING : it doesn't create the across-subject parcellation map nor the gradient map.
    """
    for s in range(18,19): # TODO : customize the range
        print('='*50)
        print('Processing subject : ', s)
        print('='*50)
        subject = f"{s:02d}"
        # Path to the subject directory
        subj_dir = dataset_dir + r"\sub-" + subject
        # Paths : TODO : to customize with your paths
        path_func = subj_dir + r"\func\wsraPPS-FACE_S" + subject + r"_005_Rest.nii"
        path_midthickness_r = subj_dir + r"\sub" + subject + r"_freesurfer\surf\rh.midthickness.32k.surf.gii"
        path_midthickness_l = subj_dir + r"\sub" + subject + r"_freesurfer\surf\lh.midthickness.32k.surf.gii"
        path_brain_mask = subj_dir + r"\sub" + subject + r"_freesurfer\mri\brainmask.mgz"
        
        single_subj_parcellation_native(subj_dir, 
                                    path_func, 
                                    path_midthickness_l, 
                                    path_midthickness_r,
                                    path_brain_mask)
        print('='*50)
        print('Success Parcellation of Subject : ', s)
        print('='*50)

# Run for the dataset1 
# multi_subj_parcellation(dataset_dir = r"D:\DATA_min_preproc\dataset_study1")
# Run for the dataset2
# multi_subj_parcellation(dataset_dir = r"D:\DATA_min_preproc\dataset_study2\sub-")


# Compute the group parcellation map
def group_parcellation(path_list):
    """From a list of outdir path, compute the average parcellation map. And then perform watershedby flooding.

    Args:
        path_list (_type_): _description_
    """
    # Load all the labels
    labels_list = []
    for path in path_list:
        labels = load_labels(path)
        labels_list.append(labels)
    labels_array = np.array(labels_list)
    # Compute the average
    average_labels = np.mean(labels_array, axis=0)
    # Perform watershed by flooding
    graph = build_mesh_graph(faces)
    edge_map = np.zeros_like(average_labels)
    for i in range(average_labels.shape[0]):
        edge_map[i] = np.mean([average_labels[j] for j in graph[i]], axis=0)
    labels = watershed_by_flooding(graph, edge_map)
    return labels


#%%

if __name__ == "__main__":
    print("Load and Test the results")
    
    # path for the dataset 2
    # path_midthickness = "D:\DATA_min_preproc\dataset_study2\sub-04\sub04_freesurfer\surf\lh.midthickness.32k.surf.gii"
    # path_midthickness_inflated = "D:\DATA_min_preproc\dataset_study2\sub-04\sub04_freesurfer\surf\lh.midthickness.inflated.32k.surf.gii"
    # path_gradient_values = "D:\DATA_min_preproc\dataset_study2\sub-04\outputs_surface\gradient_map\lh_gradient_map_20250106104701.npy"
    # path_parcels_values = "D:\DATA_min_preproc\dataset_study2\sub-04\outputs_surface\labels\lh_labels_20250106104701.npy"
    
    # Path for the dataset 1
    path_midthickness = "D:\DATA_min_preproc\dataset_study1\sub-03\sub03_freesurfer\surf\lh.midthickness.32k.surf.gii"
    path_midthickness_inflated = "D:\DATA_min_preproc\dataset_study1\sub-03\sub03_freesurfer\surf\lh.midthickness.inflated.32k.surf.gii"
    path_gradient_values = "D:\DATA_min_preproc\dataset_study1\sub-03\outputs_surface\gradient_map\lh_gradient_map_20250106155120.npy"
    path_parcels_values = "D:\DATA_min_preproc\dataset_study1\sub-03\outputs_surface\labels\lh_labels_20250106155121.npy"
    
    gii = nib.load(path_midthickness)
    coords = gii.darrays[0].data  # shape: (N_vertices, 3)
    faces = gii.darrays[1].data   # shape: (N_faces, 3)
    
    gradient_values = load_gradient_map(path_gradient_values)
    parcels_values = load_labels(path_parcels_values)
    
    # visualize_brain_surface(coords, faces, gradient_values, title='Gradient Map')
    visualize_brain_surface(coords, faces, parcels_values, title='Parcellation Map')
    
    





#%% Full pipeline old code

# vol_fmri_img, resampled_mask_img, affine = load_volume_data(path_func,
#                                                               path_brain_mask)

# surf_fmri_l, surf_fmri_r = fmri_vol2surf(vol_fmri_img, 
#                                         path_midthickness_l, 
#                                         path_midthickness_r)

# gii = nib.load(path_midthickness_l)
# coords = gii.darrays[0].data  # shape: (N_vertices, 3)
# faces = gii.darrays[1].data   # shape: (N_faces, 3)
# graph = build_mesh_graph(faces)

# path_midthickness_l_inflated = subj_dir + r"\func\lh.midthickness.inflated.32k.surf.gii"
# gii_inflated = nib.load(path_midthickness_l_inflated)
# coords_inflated = gii_inflated.darrays[0].data  # shape: (N_vertices, 3)
# faces_inflated = gii_inflated.darrays[1].data   # shape: (N_faces, 3)


# #%% Compute the similarity matrix
# # RSFC_matrix = compute_RSFC_matrix(surf_fmri_l)
# similarity_matrix = compute_similarity_matrix(surf_fmri_l, 
#                                               vol_fmri_img,
#                                               resampled_mask_img,
#                                               n_modes=179)
# sim_map = similarity_matrix[34,:]
# visualize_brain_surface(coords, faces, sim_map)
# # sim_matrx = compute_similarity_matrix(RSFC_matrix) # maybe redundant
# # Visualize inflated surface sim map
# # path_midthickness_l_inflated = subj_dir + r"\func\lh.midthickness.inflated.32k.surf.gii"
# # gii_inflated = nib.load(path_midthickness_l_inflated)
# # coords_inflated = gii_inflated.darrays[0].data  # shape: (N_vertices, 3)
# # faces_inflated = gii_inflated.darrays[1].data   # shape: (N_faces, 3)
# # visualize_brain_surface(coords_inflated, faces_inflated, sim_map)

# #%% plot the smoothed similarity map
# sim_matrix_smooothed = smooth_surface(faces,
#                                       similarity_matrix, 
#                                       iterations=5)
# visualize_brain_surface(coords, faces, sim_matrix_smooothed[34,:])
# # plotting.plot_surf_stat_map(
# #     path_midthickness_l,
# #     stat_map=sim_map_smooothed,
# #     hemi='left',
# #     view='lateral',
# #     title='sim_map_smooothed',
# #     colorbar=True,
# #     cmap='coolwarm'
# # )
# #%% plot the gradient map
# gradients = compute_gradients(graph,
#                               sim_matrix_smooothed,
#                               skip=1000)
# # load the gradient map
# # gradients = load_gradient_map("D:\DATA_min_preproc\dataset_study2\sub-01\outputs_surface\gradient_map\gradient_map_20250104163839.npy")
# visualize_brain_surface(coords, faces, gradients)
# save_gradient_map(gradients, subj_dir + r"\outputs_surface\gradient_map")
# # plotting.plot_surf_stat_map(
# #     path_midthickness_l,
# #     stat_map=gradients,
# #     hemi='left',
# #     view='lateral',
# #     title='Gradient from graph',
# #     colorbar=True,
# #     cmap='coolwarm'
# # )

# #%% smooth the gradient map
# gradient_smoothed = smooth_surface(faces,
#                                    gradients,
#                                    iterations=5)
# #%% plot the edge map
# labels = watershed_by_flooding(graph, gradient_smoothed)
# visualize_brain_surface(coords, faces, labels)
# # plotting.plot_surf_stat_map(
# #     path_midthickness_l,
# #     stat_map=edge_map*1,
# #     hemi='left',
# #     view='lateral',
# #     title='Edge Map',
# #     colorbar=True,
# #     cmap='coolwarm'
# # )