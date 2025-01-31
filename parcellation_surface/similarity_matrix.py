#%%
import os
from datetime import datetime
from typing import Literal
import numpy as np
import nibabel as nib

from preprocessing_surface import fmri_to_spatial_modes, load_volume_data
from visualization import visualize_brain_surface
from gradient import build_mesh_graph


def compute_RSFC_matrix(surf_fmri):
    """
    Compute the similarity matrix between the surface data and the volume data
    """
    print("Computing the similarity matrix...")
    print("Shape of the surface data: ", surf_fmri.shape)
    # Compute the correlation between the surface data and the volume data
    corr = np.corrcoef(surf_fmri) # TODO : complexify the case
    # r-z transform
    RSFC_matrix = np.arctanh(corr)
    # Remove inf and nans values
    RSFC_matrix[np.isinf(RSFC_matrix)] = 0
    RSFC_matrix[np.isnan(RSFC_matrix)] = 0 # Deal with NaN values
    

    return RSFC_matrix

# def compute_corr(surf_fmri, vol_fmri): TODO : remove if not used
#     """
#     Compute the correlation between the surface data and the volume data
#     """
#     print("Computing the correlation between the surface data and the volume data...")
#     print("Shape of the surface data: ", surf_fmri.shape)
#     print("Shape of the volume data: ", vol_fmri.shape)
#     # Compute the correlation between the surface data and the volume data
#     corr = np.corrcoef(surf_fmri, vol_fmri) # TODO : complexify the case
#     corr[np.isnan(corr)] = 0 # Deal with NaN values
#     # r-z transform
#     RSFC_matrix = np.arctanh(corr)
#     # Remove inf values
#     RSFC_matrix[np.isinf(RSFC_matrix)] = 0
    
#     return RSFC_matrix

def compute_similarity_matrix(surf_fmri,
                              preproc_vol_fmri_img,
                              resampled_mask_img,
                              n_modes=179):
    """
    Calculate the pairwise similarity matrix for the given dataset.

    This function computes a similarity matrix where each element [i, j] represents
    the similarity between the ith and jth samples in the input data. The similarity
    metric used is Pearson correlation.

    Args:
        surf_fmri (np.ndarray): 
            A NORMALIZED (mena=0,var=1) 2D NumPy array of shape (n_samples, n_features) representing surface fMRI data.
        preproc_vol_fmri_img (str or np.ndarray): 
            The preprocessed volumetric fMRI image. This can be a file path or a NumPy array.
        resampled_mask_img (str or np.ndarray): 
            The resampled mask image corresponding to the fMRI data. This can be a file path or a NumPy array.
        n_modes (int, optional): 
            The number of spatial modes to extract. Defaults to 179.

    Returns:
        np.ndarray: 
            A 2D NumPy array of shape (n_samples, n_samples) representing the similarity
            scores between each pair of samples.
    """
    print("Computing the similarity matrix...")
    # Get the spatial modes (np.array) (noramlized)
    spatial_modes = fmri_to_spatial_modes(preproc_vol_fmri_img, 
                                          resampled_mask_img,
                                          n_modes=n_modes)
    # Put into float32
    spatial_modes = spatial_modes.astype(np.float32)
    surf_fmri = surf_fmri.astype(np.float32)
    # Compute the correlation between the surface data and the volume data
    corr = np.dot(surf_fmri, spatial_modes.T) # both need to be normalized
    corr[np.isnan(corr)] = 0 # Deal with NaN values
    corr = np.clip(corr, -1.0, 1.0) # Clip values to the range [-1, 1] to account for numerical errors
    corr = np.arctanh(corr)
    corr[np.isinf(corr)] = 0 # Replace inf values with 0
    
    # compute the similarity matrix
    similarity_matrix = np.corrcoef(corr)
    # Remove inf and nans values
    similarity_matrix[np.isinf(similarity_matrix)] = 0
    similarity_matrix[np.isnan(similarity_matrix)] = 0 # Deal with NaN values
    
    print("Similarity matrix computed ! : shape : ", similarity_matrix.shape)

    return similarity_matrix


def save_similartiy_matrix(similarity_matrix,
                           output_dir,
                           hemisphere: Literal["lh", "rh"]
                           ) -> None:
    """Save the similarity matrix into a .npy file
    Args:
        gradient_map (2d np.array): the similartiy matrix (n_vertex, n_vertex) same order as coords
        output_dir (string): dir for sim map output
        hemisphere (strinf): the hemisphere of the surface data
    """
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    path = output_dir + f"\{hemisphere}similarity_matrix_{time}.npy"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, similarity_matrix)

def load_similarity_matrix(path):
    """
    Load the similarity matrix
    """
    similarity_matrix = np.load(path)
    return similarity_matrix

#%% Run the code

if __name__ == "__main__":
    
    surface_path = r"D:\DATA_min_preproc\dataset_study1\fsaverage6\surf\lh.white"
    surface_path = r"D:\DATA_min_preproc\dataset_study1\fsaverage6\surf\lh.inflated"
    # TODO : WARNING CUSTOMIZE THE PATH BASED ON YOUR DATA
    subject = "01"
    hemisphere = "lh"
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
    
    print("Computing the similarity matrix...")
    # Normalized the fmri data and extract the spatial modes
    similarity_matrix = compute_similarity_matrix(surf_fmri, 
                                                vol_fmri,
                                                resampled_mask,
                                                n_modes=380)
    

#%%

#####################################################
# # Compute the gradient
# def build_mesh_graph(faces):
#     """
#     Build a graph from mesh faces for neighbor lookup.
    
#     faces: ndarray of shape (n_faces, 3)
    
#     Returns:nnew
#         graph: networkx.Graph object
#     """
#     graph = nx.Graph()
#     for face in faces:
#         for i in range(3):
#             v1 = face[i]
#             v2 = face[(i + 1) % 3]
#             graph.add_edge(v1, v2)
#     return graph

# def compute_gradient(graph, stat_map):
#     """
#     Compute the gradient magnitude at each vertex based on similarity map.
    
#     similarity_map: ndarray of shape (n_vertices,)
#     graph: networkx.Graph object
    
#     Returns:
#         gradients: ndarray of shape (n_vertices,)
#     """
#     gradients = np.zeros_like(stat_map)
#     for vertex in graph.nodes:
#         neighbors = list(graph.neighbors(vertex))
#         if len(neighbors) == 0:
#             continue
#         # Compute the difference between the vertex and its neighbors
#         differences = stat_map[neighbors] - stat_map[vertex]
#         gradients[vertex] = np.sqrt(np.sum(differences ** 2))
#     return gradients

# def compute_gradients_full(graph, similarity_matrix):
#     """Compute the gradients of the similarty matrix on a mesh surface.
#     """
#     gradients = np.zeros_like(similarity_matrix[0,:])
#     for stat_map in similarity_matrix:
#         gradient = np.zeros_like(stat_map)
#         for vertex in graph.nodes:
#             neighbors = list(graph.neighbors(vertex))
#             if len(neighbors) == 0:
#                 continue
#             # Compute the difference between the vertex and its neighbors
#             differences = stat_map[neighbors] - stat_map[vertex]
#             gradient[vertex] = np.sqrt(np.sum(differences ** 2))
#         gradients += gradient
#     return gradients

# def compute_gradients(graph, similarity_matrix, skip=10):
#     """Compute the gradients of the similarty matrix on a mesh surface.
#     """
#     gradients = np.zeros_like(similarity_matrix[0,:])
#     for idx_map in range(0,similarity_matrix.shape[0], skip):
#         stat_map = similarity_matrix[idx_map,:]
#         gradient = np.zeros_like(stat_map)
#         for vertex in graph.nodes:
#             neighbors = list(graph.neighbors(vertex))
#             if len(neighbors) == 0:
#                 continue
#             # Compute the difference between the vertex and its neighbors
#             differences = stat_map[neighbors] - stat_map[vertex]
#             gradient[vertex] = np.sqrt(np.sum(differences ** 2))
#         gradients += gradient
#     return gradients

# # Non-maxima suppression
# def non_maxima_suppression(gradient_map, graph, min_neighbors=4):
#     """
#     Apply non-maxima suppression to identify edge vertices.
    
#     gradient_map: ndarray of shape (n_vertices,)
#     graph: networkx.Graph object
#     min_neighbors: int, number of non-adjacent maxima required
    
#     Returns:
#         edge_map: ndarray of shape (n_vertices,), boolean
#     """
#     edge_map = np.zeros_like(gradient_map, dtype=bool)
#     for vertex in graph.nodes:
#         neighbors = list(graph.neighbors(vertex))
#         if len(neighbors) < min_neighbors:
#             continue
#         # Check if current vertex is a local maximum
#         is_max = True
#         count = 0
#         for neighbor in neighbors:
#             if gradient_map[vertex] <= gradient_map[neighbor]:
#                 is_max = False
#                 break
#             count += 1
#             if count >= min_neighbors:
#                 break
#         if is_max and count >= min_neighbors:
#             edge_map[vertex] = True
#     return edge_map



# #%%


# if __name__ == "__main__":
#     vol_fmri_img, resampled_mask_img, affine = load_volume_data(path_func,
#                                                                   path_brain_mask)

#     surf_fmri_l, surf_fmri_r = fmri_vol2surf(vol_fmri_img, 
#                                             path_midthickness_l, 
#                                             path_midthickness_r)

#     gii = nib.load(path_midthickness_l)
#     coords = gii.darrays[0].data  # shape: (N_vertices, 3)
#     faces = gii.darrays[1].data   # shape: (N_faces, 3)
#     graph = build_mesh_graph(faces)
    
#     path_midthickness_l_inflated = subj_dir + r"\func\lh.midthickness.inflated.32k.surf.gii"
#     gii_inflated = nib.load(path_midthickness_l_inflated)
#     coords_inflated = gii_inflated.darrays[0].data  # shape: (N_vertices, 3)
#     faces_inflated = gii_inflated.darrays[1].data   # shape: (N_faces, 3)
    

#     #%% Compute the similarity matrix
#     # RSFC_matrix = compute_RSFC_matrix(surf_fmri_l)
#     similarity_matrix = compute_similarity_matrix(surf_fmri_l, 
#                                                   vol_fmri_img,
#                                                   resampled_mask_img,
#                                                   n_modes=179)
#     sim_map = similarity_matrix[34,:]
#     visualize_brain_surface(coords, faces, sim_map)
#     # sim_matrx = compute_similarity_matrix(RSFC_matrix) # maybe redundant
#     # Visualize inflated surface sim map
#     # path_midthickness_l_inflated = subj_dir + r"\func\lh.midthickness.inflated.32k.surf.gii"
#     # gii_inflated = nib.load(path_midthickness_l_inflated)
#     # coords_inflated = gii_inflated.darrays[0].data  # shape: (N_vertices, 3)
#     # faces_inflated = gii_inflated.darrays[1].data   # shape: (N_faces, 3)
#     # visualize_brain_surface(coords_inflated, faces_inflated, sim_map)

#     #%% plot the smoothed similarity map
#     sim_matrix_smooothed = smooth_surface(faces,
#                                           similarity_matrix, 
#                                           iterations=5)
#     visualize_brain_surface(coords, faces, sim_matrix_smooothed[34,:])
#     # plotting.plot_surf_stat_map(
#     #     path_midthickness_l,
#     #     stat_map=sim_map_smooothed,
#     #     hemi='left',
#     #     view='lateral',
#     #     title='sim_map_smooothed',
#     #     colorbar=True,
#     #     cmap='coolwarm'
#     # )
#     #%% plot the gradient map
#     gradients = compute_gradients(graph,
#                                   sim_matrix_smooothed,
#                                   skip=10)
#     visualize_brain_surface(coords, faces, gradients)
    
#     save_gradient_map(gradients, subj_dir + r"\outputs_surface\gradient_map")
#     # plotting.plot_surf_stat_map(
#     #     path_midthickness_l,
#     #     stat_map=gradients,
#     #     hemi='left',
#     #     view='lateral',
#     #     title='Gradient from graph',
#     #     colorbar=True,
#     #     cmap='coolwarm'
#     # )
    
#     # plot the gradient map
#     # gradients = compute_gradient_magnitudes(coords, faces, sim_map)
#     # visualize_brain_surface(coords, faces, gradients)
#     # plotting.plot_surf_stat_map(
#     #     path_midthickness_l,
#     #     stat_map=gradients,
#     #     hemi='left',
#     #     view='lateral',
#     #     title='Gradient from coords faces values',
#     #     colorbar=True,
#     #     cmap='coolwarm'
#     # )
    
#     #%% plot the edge map
#     edge_map = non_maxima_suppression(gradients,
#                                       graph,
#                                       min_neighbors=3)
#     edge_map = edge_map*1
#     visualize_brain_surface(coords, faces, edge_map)
#     # plotting.plot_surf_stat_map(
#     #     path_midthickness_l,
#     #     stat_map=edge_map*1,
#     #     hemi='left',
#     #     view='lateral',
#     #     title='Edge Map',
#     #     colorbar=True,
#     #     cmap='coolwarm'
#     # )
# # %% 



# gradients = compute_gradient(similarity_matrix[12,:],
#                              graph)


# edge_map_ = non_maxima_suppression(gradients,
#                                       graph,
#                                    min_neighbors=3)
# edge_map_ = edge_map_*1


# edge_map = np.zeros_like(edge_map_)
# for i in range(200):
#     smoothed_map = smooth_surface(faces,
#                                 similarity_matrix[100*i,:], 
#                                 iterations=3)
#     gradients = compute_gradient(similarity_matrix[100*i,:], graph)
#     edge_map_ = non_maxima_suppression(gradients,
#                                       graph,
#                                       min_neighbors=3)
#     edge_map += edge_map_*1
# %%
