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

def compute_gradients(graph, stat_map):
    """
    Compute the gradient magnitude at each vertex based on similarity map.
    
    stat_map: ndarray of shape (n_vertices,)
    graph: networkx.Graph object
    
    Returns:
        gradients: ndarray of shape (n_vertices,)
    """
    gradients = np.zeros_like(stat_map)
    for vertex in graph.nodes:
        neighbors = list(graph.neighbors(vertex))
        if len(neighbors) == 0:
            continue
        # Compute the difference between the vertex and its neighbors
        differences = stat_map[neighbors] - stat_map[vertex]
        gradients[vertex] = np.sqrt(np.sum(differences ** 2))
    return gradients




def compute_full_gradient_map(faces, similarity_matrix):
    graph = build_mesh_graph(faces)
    smoothed_sim_matrix = smooth_similarity_matrix_graph(graph, similarity_matrix)
    n_vertex = len(smoothed_sim_matrix.shape[0])
    for i in range(n_vertex):
        gradient = compute_gradients(smoothed_sim_matrix[i], graph)
        if i == 0:
            gradient_map = gradient
        else:
            gradient_map = np.vstack((gradient_map, gradient))
            
    return gradient_map


def compute_gradient_magnitudes(faces, coords, values):
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
    values : numpy.ndarray, shape (n_vertices,)
        Statistical values (e.g., t-scores, z-scores) associated with each vertex.
    iterations : int, optional (default=2)
        Number of smoothing iterations to perform.

    Returns
    -------
    smoothed_map : numpy.ndarray, shape (n_vertices,)
        Smoothed statistical values on the surface mesh.

    Raises
    ------
    ValueError
        If the number of nodes in the graph does not match the length of `values`.

    Notes
    -----
    - This function performs smoothing by averaging each vertex's value with its immediate neighbors.
    - Increasing the number of iterations results in more extensive smoothing.
    - The function leverages sparse matrix operations for efficiency.

    Example
    -------
    >>> import numpy as np
    >>> import networkx as nx
    >>> # Create a simple graph with 4 nodes
    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 1), (0, 2), (2, 3)])
    >>> values = np.array([1.0, 2.0, 3.0, 4.0])
    >>> smoothed = smooth_surface_graph(G, values, iterations=1)
    >>> print(smoothed)
    [2.5 2.  2.33333333 2.0]
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

    # Initialize smoothed_map
    smoothed_map = values.copy()

    # Perform iterative smoothing
    for _ in range(iterations):
        smoothed_map = smoothing_operator.dot(smoothed_map)

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


