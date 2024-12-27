# This file contians methods for the computation of the gradient
# of a similarity map on a mesh surface. The gradient is computed
# using the finite difference method.

#%%
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, coo_matrix
from similarity_matrix import *
from preprocessing_surface import *

def compute_gradient_magnitudes( faces, coords, values):
    """
    Compute per-vertex gradient magnitudes of a scalar field on a triangular mesh.

    Parameters
    ----------
    coords : np.ndarray, shape (n_coords, 3)
        3D coordinates of each vertex.
    faces : np.ndarray, shape (n_faces, 3)
        Triangle faces, each a triplet of vertex indices.
    values : np.ndarray, shape (n_coords,)
        Scalar value (e.g. similarity) at each vertex.

    Returns
    -------
    grad_magnitudes : np.ndarray, shape (n_coords,)
        The magnitude of the spatial gradient at each vertex.
    """

    n_vertices = coords.shape[0]
    
    # Accumulators for gradients at each vertex
    grad_accum = np.zeros((n_vertices, 3), dtype=np.float64)
    face_count = np.zeros(n_vertices, dtype=np.int32)

    for face in faces:
        i0, i1, i2 = face
        x0, x1, x2 = coords[i0], coords[i1], coords[i2]
        
        f0, f1, f2 = values[i0], values[i1], values[i2]
        
        # Edges of the triangle
        e1 = x1 - x0
        e2 = x2 - x0
        
        # Face normal and area
        n = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(n)
        
        if area < 1e-12:
            # Degenerate face (very small area) - skip to avoid division by zero
            continue

        # Compute gradient on this face in 3D
        # (d1, d2) = (f1 - f0, f2 - f0)
        d1 = f1 - f0
        d2 = f2 - f0

        # Formula for gradient of a linear function on a triangle in 3D
        # grad_face = ( d1 * (n x e2) + d2 * (e1 x n ) ) / (2 * area * ||n|| ) * ||n|| 
        # Simplifies to: 
        grad_face = ((d1 * np.cross(n, e2)) + (d2 * np.cross(e1, n))) / (2.0 * area * np.linalg.norm(n)) * np.linalg.norm(n)
        #
        # But we can fold the norm(n) in or out in different ways. An equivalent simpler formula is:
        # grad_face = ( d1 * cross(n, e2) + d2 * cross(e1, n) ) / (2 * area^2 ) 
        #
        # For numerical stability, you may see slightly different but equivalent forms.

        # Accumulate this face gradient into each vertex of the face
        grad_accum[i0] += grad_face
        grad_accum[i1] += grad_face
        grad_accum[i2] += grad_face
        
        face_count[i0] += 1
        face_count[i1] += 1
        face_count[i2] += 1

    # Average the accumulated gradients
    # Prevent division by zero for isolated vertices (if any)
    valid_mask = face_count > 0
    grad_accum[valid_mask] /= face_count[valid_mask, None]

    # Finally, compute gradient magnitude
    grad_magnitudes = np.linalg.norm(grad_accum, axis=1)
    return grad_magnitudes




def smooth_surface_stat_map(faces, coords, values, iterations=2):
    """
    Smooth a surface-based fMRI statistical map using neighborhood averaging.
    
    Parameters:
    ----------
    faces : array_like, shape (n_faces, 3)
        Indices of vertices forming each triangular face of the mesh.
        
    coords : array_like, shape (n_vertices, 3)
        3D coordinates of each vertex in the mesh.
        
    values : array_like, shape (n_vertices,)
        Statistical values (e.g., t-scores, z-scores) associated with each vertex.
        
    iterations : int, optional (default=2)
        Number of smoothing iterations to perform.
        
    Returns:
    -------
    smoothed_map : numpy.ndarray, shape (n_vertices,)
        Smoothed statistical values on the surface mesh.
        
    Notes:
    -----
    - This function performs smoothing by averaging each vertex's value with its immediate neighbors.
    - Increasing the number of iterations results in more extensive smoothing.
    - The function leverages sparse matrix operations for efficiency.
    
    Example:
    -------
    >>> import numpy as np
    >>> # Example mesh with 4 vertices and 2 faces
    >>> faces = np.array([[0, 1, 2],
    ...                   [0, 2, 3]])
    >>> coords = np.array([[0, 0, 0],
    ...                    [1, 0, 0],
    ...                    [0, 1, 0],
    ...                    [0, 0, 1]])
    >>> values = np.array([1, 2, 3, 4])
    >>> smoothed = smooth_surface_values(faces, coords, values, iterations=1)
    >>> print(smoothed)
    [ (2+3)/2, (1+3)/2, (1+2+4)/3, (1+3)/2 ] = [2.5, 2.0, 2.333..., 2.0]
    """

    # Ensure inputs are numpy arrays
    faces = np.asarray(faces)
    coords = np.asarray(coords)
    values = np.asarray(values)
    
    n_vertices = coords.shape[0]
    
    # Step 1: Extract unique edges from faces
    edges = set()
    for face in faces:
        v1, v2, v3 = face
        edges.add(tuple(sorted((v1, v2))))
        edges.add(tuple(sorted((v2, v3))))
        edges.add(tuple(sorted((v1, v3))))
    
    # Convert edges to row and column indices
    row = []
    col = []
    for edge in edges:
        row.extend([edge[0], edge[1]])  # Add both directions
        col.extend([edge[1], edge[0]])
    
    data = np.ones(len(row), dtype=np.float32)
    
    # Step 2: Create sparse adjacency matrix
    adjacency = coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices)).tocsr()
    
    # Step 3: Compute degree of each vertex
    degree = adjacency.sum(axis=1).A1  # Convert to 1D array
    degree[degree == 0] = 1  # Prevent division by zero
    
    # Step 4: Create normalization matrix (W = D^-1)
    W = csr_matrix((1.0 / degree, (np.arange(n_vertices), np.arange(n_vertices))), shape=(n_vertices, n_vertices))
    
    # Step 5: Define the smoothing operator (W * adjacency)
    smoothing_operator = W.dot(adjacency)
    
    # Step 6: Initialize smoothed_map
    smoothed_map = values.copy()
    
    # Step 7: Perform iterative smoothing
    for _ in range(iterations):
        smoothed_map = smoothing_operator.dot(smoothed_map)
    
    return smoothed_map


def save_surface_map(coords, faces, values, output_path):
    """
    Save a scalar map on a surface mesh to a GIFTI file.
    
    Parameters:
    ----------
    coords : array_like, shape (n_vertices, 3)
        3D coordinates of each vertex in the mesh.
        
    faces : array_like, shape (n_faces, 3)
        Indices of vertices forming each triangular face of the mesh.
        
    values : array_like, shape (n_vertices,)
        Scalar values to save on the mesh.
        
    output_path : str
        Path to save the GIFTI file.
        
    Notes:
    -----
    - The function creates a GIFTI file with the specified scalar values on the mesh.
    - The output file can be visualized using tools like Connectome Workbench or MRICloud.
    """
    
    # Ensure inputs are numpy arrays
    coords = np.asarray(coords)
    faces = np.asarray(faces)
    values = np.asarray(values)
    
    # Create a GIFTI image
    gii = nib.gifti.GiftiImage()
    
    # Add the coordinates
    gii.add_gifti_data_array(nib.gifti.GiftiDataArray(coords))
    
    # Add the faces
    gii.add_gifti_data_array(nib.gifti.GiftiDataArray(faces))
    
    # Add the scalar values
    gii.add_gifti_data_array(nib.gifti.GiftiDataArray(values))
    
    # Save the GIFTI image
    nib.save(gii, output_path)







from nibabel.cifti2 import (
    Cifti2Image,
    Cifti2Matrix,
    Cifti2MatrixIndicesMap,
    Cifti2BrainModel
)
import nibabel as nb
import nibabel.cifti2 as ci
def save_lh_corr_as_dconn(corr_matrix, output_dconn, structure='CIFTI_STRUCTURE_CORTEX_LEFT'):
    """
    Save an (N x N) correlation matrix to a .dconn.nii for the LEFT hemisphere.
    Compatible with older NiBabel versions where 'Cifti2MatrixIndicesMap' 
    must be called with positional arguments.
    
    Parameters
    ----------
    corr_matrix : (N, N) ndarray
        Correlation (or connectivity) matrix.
    output_dconn : str
        Output .dconn.nii filename.
    structure : str
        Brain structure name (defaults to CIFTI_STRUCTURE_CORTEX_LEFT).
    """
    N = corr_matrix.shape[0]
    assert corr_matrix.shape == (N, N), "Matrix must be NxN."

    # 1) Create top-level Cifti2Matrix container
    cifti_matrix = ci.Cifti2Matrix()

    # 2) Create ROW dimension map => BRAIN_MODELS
    # In older NiBabel, you pass:
    #   - the map type as the 1st positional argument
    #   - the dimension list as the 2nd positional argument
    #   - then named_maps/brain_models as keyword args
    row_bm = ci.Cifti2BrainModel(
        index_offset=0,
        index_count=N,
        model_type='CIFTI_MODEL_TYPE_SURFACE',
        n_surface_vertices=N,
        brain_structure=structure
    )
    row_map = ci.Cifti2MatrixIndicesMap(
        'CIFTI_INDEX_TYPE_BRAIN_MODELS',     # map type
        [0],                                 # applies_to_matrix_dimension = rows
        brain_models=[row_bm]               # attach the BrainModel list
    )

    # 3) Create COLUMN dimension map => BRAIN_MODELS
    col_bm = ci.Cifti2BrainModel(
        index_offset=0,
        index_count=N,
        model_type='CIFTI_MODEL_TYPE_SURFACE',
        surface_number_of_vertices=N,
        brain_structure=structure
    )
    col_map = ci.Cifti2MatrixIndicesMap(
        'CIFTI_INDEX_TYPE_BRAIN_MODELS',     # map type
        [1],                                 # applies_to_matrix_dimension = columns
        brain_models=[col_bm]
    )

    # 4) Add row_map and col_map to the main matrix
    cifti_matrix.extend([row_map, col_map])

    # 5) Create the Cifti2Image with your NxN data
    img = ci.Cifti2Image(dataobj=corr_matrix, header=cifti_matrix)

    # 6) Save
    img.to_filename(output_dconn)
    print(f"Saved {N}x{N} LH correlation matrix to {output_dconn}")
    
    
    
if __name__ == "__main__":
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
    
    # Load the surface data
    surf_l = nib.load(path_midthickness_l)

    coords_l, faces_l = surf_l.darrays[0].data, surf_l.darrays[1].data
    
    # Load the volume data
    preproc_vol_fmri_img, resampled_mask_img = preprocess_volume_data(path_func, path_brain_mask)
    surf_fmri_l, surf_fmri_r = fmri_vol2surf(preproc_vol_fmri_img, 
                                            path_midthickness_l, 
                                            path_midthickness_r)

    gii = nib.load(path_midthickness_l)
    coords = gii.darrays[0].data  # shape: (N_vertices, 3)
    faces = gii.darrays[1].data   # shape: (N_faces, 3)
    
    # Compute the Adjacency matrix from the mesh faces
    # adjacency_matrix = build_adjacency_matrix(faces, len(coords))
    
    #%% Compute the similarity matrix
    RSFC_matrix = compute_RSFC_matrix(surf_fmri_l)
    
    # Smooth the RSFC matrix
    RSFC_matrix_smoothed = np.zeros_like(RSFC_matrix)
    for i in range(len(RSFC_matrix)):
        RSFC_matrix_smoothed[i] = smooth_surface_stat_map(faces, coords, RSFC_matrix[i], iterations=2)
        
    smoothed_RSFC_matrix = smooth_surface_stat_map(faces, coords, RSFC_matrix, iterations=2)
    
    #%% Compute the gradient map
    mean_grad = np.zeros_like(coords[:, 0])
    # for rsfc_map in RSFC_matrix:
    #     # statmap_grad = compute_gradients(rsfc_map, adjacency_matrix)
    #     # mean_grad = statmap_grad + mean_grad
    
    
    plotting.plot_surf_stat_map(
        path_midthickness_l,
        stat_map=mean_grad,
        hemi='left',
        view='lateral',
        title='Left Hemisphere Frequency Map',
        colorbar=True,
        cmap='coolwarm'
    )
    # save the gradient map
    mean_grad_img = nib.gifti.GiftiImage()
    mean_grad_img.add_gifti_data_array(nib.gifti.GiftiDataArray(data=mean_grad))
    nib.save(mean_grad_img, subj_dir + r"\outputs\mean_gradient_map.func.gii")
    
    
# %%
