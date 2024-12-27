#%%
import numpy as np
import nibabel as nib
from nilearn import surface, plotting
import networkx as nx

from preprocessing_surface import preprocess_volume_data, fmri_vol2surf

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
    sim_matrix = np.corrcoef(surf_fmri) # TODO : complexify the case
    # print("Shape of the correlation matrix: ", corr.shape)
    # # r-z transform
    # corr_ = np.arctanh(corr)
    ## Deal with NaN and inf values
    # corr_[np.isinf(corr_)] = 0
    # # Compute the similarity matrix (correlation of correlation)
    # sim_matrix = np.corrcoef(corr_)
    print("Shape of the sim matrix: ", sim_matrix.shape)
    
    return sim_matrix


def compute_similarity_matrix(RSFC_matrix):
    """
    Compute the similarity matrix from the RSFC matrix
    """
    print("Computing the similarity matrix...")
    print("Shape of the RSFC matrix: ", RSFC_matrix.shape)
    RSFC_matrix_to_z = np.arctanh(RSFC_matrix)
    # Replace inf values with 0
    RSFC_matrix_to_z[np.isinf(RSFC_matrix_to_z)] = 0
    
    sim_matrix = np.corrcoef(RSFC_matrix_to_z) # TODO : complexify the case
    print("Shape of the sim matrix: ", sim_matrix.shape)
    return sim_matrix



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
def non_maxima_suppression(gradient_map, graph, min_neighbors=2):
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






if __name__ == "__main__":
    from gradient import smooth_surface_stat_map
    preproc_vol_fmri_img, resampled_mask_img = preprocess_volume_data(path_func,
                                                                  path_brain_mask)

    surf_fmri_l, surf_fmri_r = fmri_vol2surf(preproc_vol_fmri_img, 
                                            path_midthickness_l, 
                                            path_midthickness_r)

    gii = nib.load(path_midthickness_l)
    coords = gii.darrays[0].data  # shape: (N_vertices, 3)
    faces = gii.darrays[1].data   # shape: (N_faces, 3)
    
    # Compute the similarity matrix
    RSFC_matrix = compute_RSFC_matrix(surf_fmri_l)
    # sim_matrx = compute_similarity_matrix(RSFC_matrix) # maybe redundant
    #%%
    graph = build_mesh_graph(faces)
    # Smooth the similarity map
    sim_map = RSFC_matrix[12312,:]
    plotting.plot_surf_stat_map(
        path_midthickness_l,
        stat_map=sim_map,
        hemi='left',
        view='lateral',
        title='sim_map',
        colorbar=True,
        cmap='coolwarm'
    )
    sim_map_smooothed = smooth_surface_stat_map(faces, coords, sim_map, iterations=2)
    plotting.plot_surf_stat_map(
        path_midthickness_l,
        stat_map=sim_map_smooothed,
        hemi='left',
        view='lateral',
        title='sim_map_smooothed',
        colorbar=True,
        cmap='coolwarm'
    )
    
    gradients = compute_gradients(RSFC_matrix[12312,:], graph)
    plotting.plot_surf_stat_map(
        path_midthickness_l,
        stat_map=gradients,
        hemi='left',
        view='lateral',
        title='Left Hemisphere Frequency Map',
        colorbar=True,
        cmap='coolwarm'
    )
    
    edge_map = non_maxima_suppression(gradients,
                                      graph,
                                      min_neighbors=2)
    plotting.plot_surf_stat_map(
        path_midthickness_l,
        stat_map=edge_map*1,
        hemi='left',
        view='lateral',
        title='Edge Map',
        colorbar=True,
        cmap='coolwarm'
    )
# %%
