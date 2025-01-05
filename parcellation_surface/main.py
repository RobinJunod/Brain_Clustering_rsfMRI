#TODO : this file is for the entire pipeline computation
#%%
import numpy as np
import nibabel as nib

from preprocessing_surface import load_volume_data, fmri_vol2surf
from similarity_matrix import compute_similarity_matrix
from smoothing import smooth_surface
from gradient import compute_gradients, build_mesh_graph, load_gradient_map,save_gradient_map
from watershed import watershed_by_flooding
from visualization import visualize_brain_surface


# Paths
SUBJECT = r"01"

subj_dir = r"D:\DATA_min_preproc\dataset_study2\sub-" + SUBJECT
path_func = subj_dir + r"\func\rwsraOB_TD_FBI_S" + SUBJECT + r"_007_Rest.nii"

path_midthickness_r = subj_dir + r"\func\rh.midthickness.32k.surf.gii"
path_midthickness_l = subj_dir + r"\func\lh.midthickness.32k.surf.gii"

path_white_r = subj_dir + r"\func\rh.white.32k.surf.gii"
path_white_l = subj_dir + r"\func\lh.white.32k.surf.gii"

path_pial_r = subj_dir + r"\func\rh.pial.32k.surf.gii"
path_pial_l = subj_dir + r"\func\lh.pial.32k.surf.gii"

path_brain_mask = subj_dir + r"\sub" + SUBJECT + r"_freesurfer\mri\brainmask.mgz"


if __name__ == "__main__":
    vol_fmri_img, resampled_mask_img, affine = load_volume_data(path_func,
                                                                  path_brain_mask)

    surf_fmri_l, surf_fmri_r = fmri_vol2surf(vol_fmri_img, 
                                            path_midthickness_l, 
                                            path_midthickness_r)

    gii = nib.load(path_midthickness_l)
    coords = gii.darrays[0].data  # shape: (N_vertices, 3)
    faces = gii.darrays[1].data   # shape: (N_faces, 3)
    graph = build_mesh_graph(faces)
    
    path_midthickness_l_inflated = subj_dir + r"\func\lh.midthickness.inflated.32k.surf.gii"
    gii_inflated = nib.load(path_midthickness_l_inflated)
    coords_inflated = gii_inflated.darrays[0].data  # shape: (N_vertices, 3)
    faces_inflated = gii_inflated.darrays[1].data   # shape: (N_faces, 3)
    

    #%% Compute the similarity matrix
    # RSFC_matrix = compute_RSFC_matrix(surf_fmri_l)
    similarity_matrix = compute_similarity_matrix(surf_fmri_l, 
                                                  vol_fmri_img,
                                                  resampled_mask_img,
                                                  n_modes=179)
    sim_map = similarity_matrix[34,:]
    visualize_brain_surface(coords, faces, sim_map)
    # sim_matrx = compute_similarity_matrix(RSFC_matrix) # maybe redundant
    # Visualize inflated surface sim map
    # path_midthickness_l_inflated = subj_dir + r"\func\lh.midthickness.inflated.32k.surf.gii"
    # gii_inflated = nib.load(path_midthickness_l_inflated)
    # coords_inflated = gii_inflated.darrays[0].data  # shape: (N_vertices, 3)
    # faces_inflated = gii_inflated.darrays[1].data   # shape: (N_faces, 3)
    # visualize_brain_surface(coords_inflated, faces_inflated, sim_map)

    #%% plot the smoothed similarity map
    sim_matrix_smooothed = smooth_surface(faces,
                                          similarity_matrix, 
                                          iterations=5)
    visualize_brain_surface(coords, faces, sim_matrix_smooothed[34,:])
    # plotting.plot_surf_stat_map(
    #     path_midthickness_l,
    #     stat_map=sim_map_smooothed,
    #     hemi='left',
    #     view='lateral',
    #     title='sim_map_smooothed',
    #     colorbar=True,
    #     cmap='coolwarm'
    # )
    #%% plot the gradient map
    gradients = compute_gradients(graph,
                                  sim_matrix_smooothed,
                                  skip=1000)
    # load the gradient map
    # gradients = load_gradient_map("D:\DATA_min_preproc\dataset_study2\sub-01\outputs_surface\gradient_map\gradient_map_20250104163839.npy")
    visualize_brain_surface(coords, faces, gradients)
    save_gradient_map(gradients, subj_dir + r"\outputs_surface\gradient_map")
    # plotting.plot_surf_stat_map(
    #     path_midthickness_l,
    #     stat_map=gradients,
    #     hemi='left',
    #     view='lateral',
    #     title='Gradient from graph',
    #     colorbar=True,
    #     cmap='coolwarm'
    # )
    
    #%% smooth the gradient map
    gradient_smoothed = smooth_surface(faces,
                                       gradients,
                                       iterations=5)
    #%% plot the edge map
    labels = watershed_by_flooding(graph, gradient_smoothed)
    visualize_brain_surface(coords, faces, labels)
    # plotting.plot_surf_stat_map(
    #     path_midthickness_l,
    #     stat_map=edge_map*1,
    #     hemi='left',
    #     view='lateral',
    #     title='Edge Map',
    #     colorbar=True,
    #     cmap='coolwarm'
    # )
# %% 