#%%
import numpy as np
import nibabel as nib
from nilearn import surface, plotting
import networkx as nx

from preprocessing_surface import load_volume_data, fmri_vol2surf, fmri_to_spatial_modes
from visualization_toolbox import visualize_brain_surface, visualize_brain_scalar
from surf_map_operations import compute_gradient_magnitudes, \
                                smooth_surface_stat_map
                        

"""
Inputs :
- fmri projected into surface space
- frmi data in volume space
- mask data in volume space

output : a similartiy matrix of shape (n, n) where n is the number of vertices in the surface
"""

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
#%%

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

def compute_corr(surf_fmri, vol_fmri):
    """
    Compute the correlation between the surface data and the volume data
    """
    print("Computing the correlation between the surface data and the volume data...")
    print("Shape of the surface data: ", surf_fmri.shape)
    print("Shape of the volume data: ", vol_fmri.shape)
    # Compute the correlation between the surface data and the volume data
    corr = np.corrcoef(surf_fmri, vol_fmri) # TODO : complexify the case
    corr[np.isnan(corr)] = 0 # Deal with NaN values
    # r-z transform
    RSFC_matrix = np.arctanh(corr)
    # Remove inf values
    RSFC_matrix[np.isinf(RSFC_matrix)] = 0
    
    return RSFC_matrix

def compute_similarity_matrix(surf_fmri,
                              preproc_vol_fmri_img,
                              resampled_mask_img,
                              n_modes=50):
    """_summary_

    Args:
        surf_fmri (np.array): nomalized surface data
        preproc_vol_fmri_img (_type_): _description_
        resampled_mask_img (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("Computing the similarity matrix...")
    # Get the spatial modes (np.array) (noramlized)
    spatial_modes = fmri_to_spatial_modes(preproc_vol_fmri_img, 
                                          resampled_mask_img,
                                          n_modes=n_modes)
    corr = np.dot(surf_fmri, spatial_modes.T)
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




# Compute the gradient
def build_mesh_graph(faces):
    """
    Build a graph from mesh faces for neighbor lookup.
    
    faces: ndarray of shape (n_faces, 3)
    
    Returns:nnew
        graph: networkx.Graph object
    """
    graph = nx.Graph()
    for face in faces:
        for i in range(3):
            v1 = face[i]
            v2 = face[(i + 1) % 3]
            graph.add_edge(v1, v2)
    return graph

def compute_gradients(similarity_map, graph):
    """
    Compute the gradient magnitude at each vertex based on similarity map.
    
    similarity_map: ndarray of shape (n_vertices,)
    graph: networkx.Graph object
    
    Returns:
        gradients: ndarray of shape (n_vertices,)
    """
    gradients = np.zeros_like(similarity_map)
    for vertex in graph.nodes:
        neighbors = list(graph.neighbors(vertex))
        if len(neighbors) == 0:
            continue
        # Compute the difference between the vertex and its neighbors
        differences = similarity_map[neighbors] - similarity_map[vertex]
        gradients[vertex] = np.sqrt(np.sum(differences ** 2))
    return gradients

# Non-maxima suppression
def non_maxima_suppression(gradient_map, graph, min_neighbors=4):
    """
    Apply non-maxima suppression to identify edge vertices.
    
    gradient_map: ndarray of shape (n_vertices,)
    graph: networkx.Graph object
    min_neighbors: int, number of non-adjacent maxima required
    
    Returns:
        edge_map: ndarray of shape (n_vertices,), boolean
    """
    edge_map = np.zeros_like(gradient_map, dtype=bool)
    for vertex in graph.nodes:
        neighbors = list(graph.neighbors(vertex))
        if len(neighbors) < min_neighbors:
            continue
        # Check if current vertex is a local maximum
        is_max = True
        count = 0
        for neighbor in neighbors:
            if gradient_map[vertex] <= gradient_map[neighbor]:
                is_max = False
                break
            count += 1
            if count >= min_neighbors:
                break
        if is_max and count >= min_neighbors:
            edge_map[vertex] = True
    return edge_map



#%%


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
    
    #%% Compute the similarity matrix
    # RSFC_matrix = compute_RSFC_matrix(surf_fmri_l)
    similarity_matrix = compute_similarity_matrix(surf_fmri_l, 
                                                  vol_fmri_img,
                                                  resampled_mask_img,
                                                  n_modes=179)
    sim_map = similarity_matrix[34,:]
    visualize_brain_surface(coords, faces, sim_map)
    # sim_matrx = compute_similarity_matrix(RSFC_matrix) # maybe redundant
    #%% plot the similarity map
    # Smooth the similarity map
    # plotting.plot_surf_stat_map(
    #     path_midthickness_l,
    #     stat_map=sim_map,
    #     hemi='left',
    #     view='lateral',
    #     title='sim_map',
    #     colorbar=True,
    #     cmap='coolwarm'
    # )
    #%% plot the smoothed similarity map
    sim_map_smooothed = smooth_surface_stat_map(faces,
                                                coords,
                                                sim_map, 
                                                iterations=3)
    visualize_brain_surface(coords, faces, sim_map_smooothed)
    #%% plotting.plot_surf_stat_map(
    #     path_midthickness_l,
    #     stat_map=sim_map_smooothed,
    #     hemi='left',
    #     view='lateral',
    #     title='sim_map_smooothed',
    #     colorbar=True,
    #     cmap='coolwarm'
    # )
    #%% plot the gradient map
    gradients = compute_gradients(sim_map_smooothed, graph)
    visualize_brain_surface(coords, faces, gradients)
    
    #%% plotting.plot_surf_stat_map(
    #     path_midthickness_l,
    #     stat_map=gradients,
    #     hemi='left',
    #     view='lateral',
    #     title='Gradient from graph',
    #     colorbar=True,
    #     cmap='coolwarm'
    # )
    
    #%% plot the gradient map
    gradients = compute_gradient_magnitudes(coords, faces, sim_map_smooothed)
    visualize_brain_surface(coords, faces, gradients)
    #%% plotting.plot_surf_stat_map(
    #     path_midthickness_l,
    #     stat_map=gradients,
    #     hemi='left',
    #     view='lateral',
    #     title='Gradient from coords faces values',
    #     colorbar=True,
    #     cmap='coolwarm'
    # )
    
    #%% plot the edge map
    edge_map = non_maxima_suppression(gradients,
                                      graph,
                                      min_neighbors=3)
    edge_map = edge_map*1
    visualize_brain_surface(coords, faces, edge_map)
    #%% plotting.plot_surf_stat_map(
    #     path_midthickness_l,
    #     stat_map=edge_map*1,
    #     hemi='left',
    #     view='lateral',
    #     title='Edge Map',
    #     colorbar=True,
    #     cmap='coolwarm'
    # )
# %% 



gradients = compute_gradients(similarity_matrix[12,:], graph)


edge_map_ = non_maxima_suppression(gradients,
                                      graph,
                                   min_neighbors=3)
edge_map_ = edge_map_*1


edge_map = np.zeros_like(edge_map_)
for i in range(200):
    smoothed_map = smooth_surface_stat_map(faces,
                                        coords,
                                        similarity_matrix[100*i,:], 
                                        iterations=3)
    gradients = compute_gradients(similarity_matrix[100*i,:], graph)
    edge_map_ = non_maxima_suppression(gradients,
                                      graph,
                                      min_neighbors=3)
    edge_map += edge_map_*1