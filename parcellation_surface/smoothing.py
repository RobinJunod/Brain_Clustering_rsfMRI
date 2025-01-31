# This file contians methods for the computation of the gradient
# of a similarity map on a mesh surface. The gradient is computed
# using the finite difference method.

#%%
import numpy as np
import nibabel as nib
import networkx as nx
from scipy.sparse import csr_matrix, coo_matrix


def smooth_surface(faces, values, iterations=5):
    """    
    Smooth a surface-based fMRI statistical map using neighborhood averaging.

    Args:
        faces : (M, 3) ndarray
            Triangles as vertex indices.
        values (np.array): 1D or 2D array of statistical values on the surface mesh. (n_vertices, n_maps)
        iterations (int, optional): smoothing intensity. Defaults to 5.

    Returns:
        np.array : 1D or 2D array of smoothed statistical values on the surface mesh. (n_vertices, n_maps)
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

from scipy.sparse import coo_matrix, csr_matrix

def smooth_surface_with_graph(graph, 
                              values, 
                              iterations=5):
    """
    Smooth a surface-based fMRI statistical map using neighborhood averaging,
    using a NetworkX graph to define adjacency.

    Args:
        graph (networkx.Graph):
            Undirected graph where each node is a vertex on the surface (labeled 0..n_vertices-1).
            Edges define adjacency (mesh connectivity).
        values (np.ndarray):
            A 1D or 2D array of shape (n_vertices,) or (n_vertices, n_maps).
        iterations (int, optional):
            Number of smoothing iterations (averaging steps). Defaults to 5.

    Returns:
        np.ndarray:
            The smoothed map, same shape as `values` (1D or 2D).
    """
    # Convert 'values' to float NumPy array
    values = np.asarray(values, dtype=np.float32)

    # Number of vertices (make sure this matches the number of nodes in the graph)
    n_vertices = values.shape[0]
    if graph.number_of_nodes() != n_vertices:
        raise ValueError(f"Graph has {graph.number_of_nodes()} nodes but 'values' has {n_vertices} vertices.")

    # 1) Build the sparse adjacency matrix (n_vertices x n_vertices) in COO format
    row = []
    col = []
    data = []
    for u, v in graph.edges():
        # Undirected => add both (u->v) and (v->u)
        row.extend([u, v])
        col.extend([v, u])
        data.extend([1.0, 1.0])

    adjacency_coo = coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
    adjacency = adjacency_coo.tocsr()  # convert to CSR for fast row operations

    # 2) Degree of each vertex (sum of adjacency row)
    degree = np.array(adjacency.sum(axis=1)).ravel()  # shape (n_vertices,)
    degree[degree == 0] = 1.0  # avoid divide-by-zero for isolated nodes

    # 3) Create the diagonal matrix W = D^-1
    W_data = 1.0 / degree
    W = csr_matrix((W_data, (range(n_vertices), range(n_vertices))), shape=(n_vertices, n_vertices))

    # 4) Smoothing operator: S = W * adjacency
    smoothing_operator = W.dot(adjacency)

    # 5) Iterative smoothing
    smoothed_map = values.copy()  # shape: (n_vertices,) or (n_vertices, n_maps)
    for _ in range(iterations):
        smoothed_map = smoothing_operator.dot(smoothed_map)

    return smoothed_map




def smooth_surface_with_graph_adjlist(graph, values, iterations=10):
    """
    Smooth a surface-based fMRI statistical map using neighborhood averaging,
    defined by a NetworkX graph. This version is more memory-efficient
    because it avoids building a full n x n adjacency matrix.

    Args:
        graph (networkx.Graph):
            Undirected graph where each node is a vertex on the surface (0..n_vertices-1).
            Edges define adjacency.
        values (np.ndarray):
            A 1D or 2D array of shape (n_vertices,) or (n_vertices, n_maps).
        iterations (int, optional):
            Number of smoothing iterations (averaging steps). Defaults to 5.

    Returns:
        np.ndarray:
            The smoothed map, same shape as `values` (1D or 2D).
    """
    values = np.asarray(values, dtype=np.float32)
    n_vertices = values.shape[0]

    if graph.number_of_nodes() != n_vertices:
        raise ValueError(
            f"Graph has {graph.number_of_nodes()} nodes but 'values' has {n_vertices} vertices."
        )

    # Build adjacency lists: adjacency_list[u] = list of neighbors of u
    adjacency_list = [[] for _ in range(n_vertices)]
    for u, v in graph.edges():
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)

    # Prepare buffers for iterative updates
    smoothed_map = values.copy()
    new_map = np.zeros_like(smoothed_map)

    for _ in range(iterations):
        if smoothed_map.ndim == 1:
            # 1D case
            for u in range(n_vertices):
                neighbors = adjacency_list[u]
                deg = len(neighbors)
                # Avoid division by zero in case of isolated nodes
                if deg > 0:
                    s = 0.0
                    for v in neighbors:
                        s += smoothed_map[v]
                    new_map[u] = s / deg
                else:
                    new_map[u] = smoothed_map[u]
        else:
            # 2D case: shape (n_vertices, n_maps)
            for u in range(n_vertices):
                neighbors = adjacency_list[u]
                deg = len(neighbors)
                if deg > 0:
                    # sum across neighbors for each map separately
                    new_map[u, :] = smoothed_map[neighbors, :].sum(axis=0) / deg
                else:
                    new_map[u, :] = smoothed_map[u, :]

        # Swap buffers instead of copying
        smoothed_map, new_map = new_map, smoothed_map

    # If we did an odd number of iterations, the "final" result is in smoothed_map.
    # If even, it's in new_map. We can handle that by returning smoothed_map
    # if we ended with an odd iteration count; or simply do:
    return smoothed_map


# Usage:
# save_gifti_stat_map(sim_map_smoothed, "gradient.func.gii")

import numpy as np
import networkx as nx
import numba
from numba import njit, prange, typed, types

@njit(parallel=True)
def _smooth_iteration_1d(adj_list_nb, smoothed_map, new_map):
    n_vertices = smoothed_map.size
    for u in prange(n_vertices):
        neighbors = adj_list_nb[u]
        deg = len(neighbors)
        if deg > 0:
            s = 0.0
            for v in neighbors:
                s += smoothed_map[v]
            new_map[u] = s / deg
        else:
            new_map[u] = smoothed_map[u]

@njit(parallel=True)
def _smooth_iteration_2d(adj_list_nb, smoothed_map, new_map):
    n_vertices, n_maps = smoothed_map.shape
    for u in prange(n_vertices):
        neighbors = adj_list_nb[u]
        deg = len(neighbors)
        if deg > 0:
            for m in range(n_maps):
                s = 0.0
                for v in neighbors:
                    s += smoothed_map[v, m]
                new_map[u, m] = s / deg
        else:
            # isolated node => copy old values
            for m in range(n_maps):
                new_map[u, m] = smoothed_map[u, m]

def smooth_surface_with_graph_adjlist_numba(graph, values, iterations=5):
    """
    Smooth a surface-based fMRI statistical map using neighborhood averaging,
    defined by a NetworkX graph. Uses Numba JIT to accelerate the iterative loop.

    Args:
        graph (networkx.Graph):
            Undirected graph where each node is a vertex on the surface (0..n_vertices-1).
            Edges define adjacency.
        values (np.ndarray):
            A 1D or 2D array of shape (n_vertices,) or (n_vertices, n_maps).
        iterations (int, optional):
            Number of smoothing iterations (averaging steps). Defaults to 5.

    Returns:
        np.ndarray: The smoothed map, same shape as `values`.
    """

    # Convert input to float32 array
    values = np.asarray(values, dtype=np.float32)
    n_vertices = values.shape[0]

    if graph.number_of_nodes() != n_vertices:
        raise ValueError(
            f"Graph has {graph.number_of_nodes()} nodes but 'values' has {n_vertices} vertices."
        )

    # -----------------------------------------------------------
    # 1) Build a Numba-typed adjacency list
    #    adjacency_list[u] = list of neighbors of u
    # -----------------------------------------------------------
    # First build a normal Python list of lists
    adjacency_list = [[] for _ in range(n_vertices)]
    for u, v in graph.edges():
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)

    # Now convert each Python list to a typed.List[int64]
    adjacency_list_nb = typed.List.empty_list(types.ListType(numba.int64))
    for u in range(n_vertices):
        nb_list = typed.List.empty_list(numba.int64)
        for neighbor in adjacency_list[u]:
            nb_list.append(neighbor)
        adjacency_list_nb.append(nb_list)

    # -----------------------------------------------------------
    # 2) Prepare buffers for iterative updates
    # -----------------------------------------------------------
    smoothed_map = values.copy()
    new_map = np.zeros_like(smoothed_map)

    # -----------------------------------------------------------
    # 3) Iterative smoothing using Numba-accelerated functions
    # -----------------------------------------------------------
    if smoothed_map.ndim == 1:
        for _ in range(iterations):
            _smooth_iteration_1d(adjacency_list_nb, smoothed_map, new_map)
            # Swap references instead of copying large arrays
            smoothed_map, new_map = new_map, smoothed_map
    else:
        for _ in range(iterations):
            _smooth_iteration_2d(adjacency_list_nb, smoothed_map, new_map)
            smoothed_map, new_map = new_map, smoothed_map

    # If we did an odd number of iterations, the final result is in smoothed_map.
    # Otherwise, it's in new_map. We always return smoothed_map after final swap.
    return smoothed_map



if __name__ == "__main__":
    pass
    
    
#%%

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


