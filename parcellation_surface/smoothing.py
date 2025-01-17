# This file contians methods for the computation of the gradient
# of a similarity map on a mesh surface. The gradient is computed
# using the finite difference method.

#%%
import numpy as np
import nibabel as nib
import networkx as nx
from scipy.sparse import csr_matrix, coo_matrix

"""
2 method for computing the gradient, one is faster using simply 
the difference between the vertex and its neighbors in a graph. The other
is more accurate and uses the formula for the gradient of a linear function
(takes the coords of the vertex).
"""
def smooth_surface(faces, values, iterations=2):
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
    values = np.asarray(values, dtype=np.float32)
    
    n_vertices = values.shape[0]
    
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

def smooth_surface_graph(
    graph: nx.Graph,
    values: np.ndarray,
    iterations: int = 2
) -> np.ndarray:
    """
    Smooth a surface-based fMRI statistical map using a NetworkX graph for neighborhood averaging.

    Parameters
    ----------
    graph : networkx.Graph
        A NetworkX graph where nodes represent vertices and edges represent adjacency.
    values : numpy.ndarray, shape (n_vertices,) or (n_surface , n_vertices)
        Statistical values (e.g., t-scores, z-scores) associated with each vertex.
    iterations : int, optional (default=2)
        Number of smoothing iterations to perform.

    Returns
    -------
    smoothed_map : numpy.ndarray, shape (n_vertices,)
        Smoothed statistical values on the surface mesh.

    """
    # Validate inputs
    if not isinstance(graph, nx.Graph):
        raise TypeError("`graph` must be a NetworkX Graph or DiGraph object.")
    
    n_vertices = len(graph)
    if n_vertices != len(values):
        raise ValueError("The number of nodes in the graph must match the length of `values`.")

    # Ensure the graph is undirected for symmetric adjacency
    if graph.is_directed():
        graph = graph.to_undirected()

    # Relabel nodes to ensure they are in the range [0, n_vertices-1]
    # This is important for constructing the adjacency matrix correctly
    mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)

    # Construct the adjacency matrix in CSR format
    adjacency_matrix = nx.to_scipy_sparse_array(
        graph, format='csr', dtype=np.float32, 
        nodelist=range(n_vertices)
    )

    # Compute the degree of each vertex
    degree = adjacency_matrix.sum(axis=1)  # Convert to 1D array
    degree[degree == 0] = 1  # Prevent division by zero

    # Create the normalization matrix (W = D^-1)
    W = csr_matrix(
        (1.0 / degree, (np.arange(n_vertices), np.arange(n_vertices))),
        shape=(n_vertices, n_vertices)
    )

    # Define the smoothing operator (W * adjacency)
    smoothing_operator = W.dot(adjacency_matrix)

    # Perform iterative smoothing
    for _ in range(iterations):
        smoothed_map = smoothing_operator.dot(values)

    return smoothed_map



from nibabel.gifti import GiftiImage, GiftiDataArray
def save_coords_faces_values_to_gii(coords, faces, values, out_gii):
    """
    Save surface geometry (coords & faces) and per-vertex 'values'
    into a single GIFTI file (.gii).

    Parameters
    ----------
    coords : np.ndarray, shape (n_vertices, 3)
        Array of x,y,z coordinates for each vertex.
    faces : np.ndarray, shape (n_faces, 3)
        Each row has the 3 vertex indices defining one triangle.
    values : np.ndarray, shape (n_vertices,) or (n_vertices, 1)
        The per-vertex data you want to store (e.g., stat or shape values).
    out_gii : str
        Path to the output GIFTI file, e.g. "my_surf.gii".
    """

    # Convert data to float32 or int32 as appropriate
    coords = coords.astype(np.float32)
    faces  = faces.astype(np.int32)
    # Ensure values are float32 (typical for surface scalar data)
    values = values.astype(np.float32)

    # 1) GIFTI array for vertex coordinates
    coords_darray = GiftiDataArray(
        data=coords,
        intent="NIFTI_INTENT_POINTSET"   # Required intent for surface vertices
    )
    # 2) GIFTI array for faces
    faces_darray = GiftiDataArray(
        data=faces,
        intent="NIFTI_INTENT_TRIANGLE"   # Required intent for surface faces
    )
    # 3) GIFTI array for per-vertex data
    #    'NIFTI_INTENT_SHAPE' or 'NIFTI_INTENT_NONE' are common choices
    values_darray = GiftiDataArray(
        data=values,
        intent="NIFTI_INTENT_SHAPE"      # Could also be 'NIFTI_INTENT_NONE', etc.
    )

    # Combine into one GiftiImage
    gifti_img = GiftiImage(darrays=[coords_darray, faces_darray, values_darray])

    # Save
    nib.save(gifti_img, out_gii)
    print(f"Saved GIFTI file with coords, faces, and values: {out_gii}")





# Usage:
# save_gifti_stat_map(sim_map_smoothed, "gradient.func.gii")



if __name__ == "__main__":
    pass
    
    
#%%

def build_adjacency(n_vertices, faces):
    neighbors = [[] for _ in range(n_vertices)]
    for (i0, i1, i2) in faces:
        neighbors[i0].append(i1)
        neighbors[i0].append(i2)
        neighbors[i1].append(i0)
        neighbors[i1].append(i2)
        neighbors[i2].append(i0)
        neighbors[i2].append(i1)
    # remove duplicates
    neighbors = [list(set(nbrs)) for nbrs in neighbors]
    return neighbors

def find_local_minima(values, neighbors):
    """
    Return a list of indices of vertices that are local minima.
    i.e., value[i] < value[nbr] for all nbr in neighbors[i].
    """
    minima = []
    for i, val_i in enumerate(values):
        is_min = True
        for nbr in neighbors[i]:
            if values[nbr] <= val_i:
                is_min = False
                break
        if is_min:
            minima.append(i)
    return minima
import heapq


def watershed_by_flooding(vertices, faces, values, neighbors=None):
    """
    Perform a watershed segmentation on 'values' defined on a surface mesh.
    Returns:
        labels : (N,) array of integers, each vertex's basin label
                 or -2 for boundary vertices, -1 for unassigned.
    """
    n_vertices = vertices.shape[0]
    if neighbors is None:
        neighbors = build_adjacency(n_vertices, faces)

    # 1) Find local minima
    minima_indices = find_local_minima(values, neighbors)

    # labels[i] = which basin index (>=0) or boundary (-2) or unassigned (-1)
    labels = np.full(n_vertices, -1, dtype=int)

    # 2) Assign each local minimum a unique label
    for basin_id, idx in enumerate(minima_indices):
        labels[idx] = basin_id

    # 3) Priority queue (value, vertex_index)
    pq = []
    # Initialize queue with local minima
    for i in minima_indices:
        heapq.heappush(pq, (values[i], i))

    # 4) Flooding
    while pq:
        val_i, i = heapq.heappop(pq)
        basin_label = labels[i]
        if basin_label < 0:
            # It's been changed since the time we pushed it in the queue
            continue
        if basin_label == -2:
            # This is already boundary => do not flood
            continue

        # Explore neighbors
        for nbr in neighbors[i]:
            nbr_label = labels[nbr]
            if nbr_label == -1:
                # unassigned => adopt i's basin
                labels[nbr] = basin_label
                heapq.heappush(pq, (values[nbr], nbr))
            elif nbr_label >= 0 and nbr_label != basin_label:
                # Conflict => boundary
                labels[i] = -2
                labels[nbr] = -2
                # Mark both as boundary => do not expand from them
                break  # stop flooding from i

    return labels


####################### EXAMPLE USAGE  #######################


# import numpy as np
# from collections import defaultdict, deque

# def build_adjacency(n_vertices, faces):
#     neighbors = defaultdict(set)
#     for (v1, v2, v3) in faces:
#         neighbors[v1].update([v2, v3])
#         neighbors[v2].update([v1, v3])
#         neighbors[v3].update([v1, v2])
#     return dict(neighbors)

# def compute_gradient_magnitude(values, neighbors):
#     n_vertices = len(values)
#     grad_map = np.zeros(n_vertices, dtype=float)
#     for v in range(n_vertices):
#         val_v = values[v]
#         nb_vals = [values[n] for n in neighbors[v]]
#         # e.g., approximate local gradient as stdev of neighbors
#         # or use max absolute difference:
#         diffs = [abs(val_v - nv) for nv in nb_vals]
#         grad_map[v] = np.max(diffs)  # or use np.max(diffs)
#     return grad_map

# def detect_edges_surface(values, faces, high_ratio=0.8, low_ratio=0.4):
#     """
#     A simplified 'Canny-like' approach on a surface:
#       1) Build adjacency
#       2) Compute gradient magnitude
#       3) Hysteresis thresholding with two thresholds

#     values: (N,) array of scalar data
#     faces: (M,3) array of triangle indices
#     high_ratio: fraction of the max gradient to define 'high' threshold
#     low_ratio: fraction of the max gradient to define 'low' threshold

#     returns edge_map: (N,) boolean array for which vertices are edges
#     """
#     n_vertices = len(values)
#     neighbors = build_adjacency(n_vertices, faces)
    
#     # 1) Compute gradient magnitude
#     grad_map = compute_gradient_magnitude(values, neighbors)
    
#     # 2) Determine thresholds
#     gmax = grad_map.max()
#     high_thr = high_ratio * gmax
#     low_thr  = low_ratio  * gmax

#     # 3) Label strong edges (above high_thr) and weak edges (between low_thr, high_thr)
#     strong_edges = np.where(grad_map >= high_thr)[0]
#     weak_edges   = np.where((grad_map >= low_thr) & (grad_map < high_thr))[0]
    
#     edge_map = np.zeros(n_vertices, dtype=bool)
#     edge_map[strong_edges] = True  # Mark strong edges

#     # 4) Hysteresis: BFS or DFS from strong edges to see if we can include weak edges
#     weak_set = set(weak_edges)  # to check membership quickly
#     queue = deque(strong_edges)
#     while queue:
#         v = queue.popleft()
#         for nb in neighbors[v]:
#             if nb in weak_set and (not edge_map[nb]):
#                 # Promote this weak edge to a 'real' edge
#                 edge_map[nb] = True
#                 queue.append(nb)
    
#     return edge_map, grad_map


