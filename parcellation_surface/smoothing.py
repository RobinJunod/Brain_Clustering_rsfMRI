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
    Smooth a surface-based fMRI statistical map using neighborhood averaging. WARNING : Memory expensive for large maps.
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


def smooth_surface_graph(graph, values, iterations=10):
    """
    Smooth a surface-based fMRI statistical map using neighborhood averaging iteratively.
    This version is more memory-efficient because it 
    avoids building a full n x n adjacency matrix.
    
    Args:
        graph (networkx.Graph):
            Undirected graph where each node is a vertex on the surface (0..n_vertices-1).
            Edges define adjacency.
        values (np.ndarray):
            A 1D or 2D array of shape (n_vertices,) or (n_vertices, n_maps).
        iterations (int, optional):
            Number of smoothing iterations (averaging steps). Defaults to 10.
    
    Returns:
        np.ndarray:
            The smoothed map, with the same shape as `values`:
              - For 1D input: (n_vertices,)
              - For 2D input: (n_vertices, n_maps)
    """
    # Ensure values are a NumPy array of float32
    values = np.asarray(values, dtype=np.float32)
    n_vertices = values.shape[0]
    
    # Validate that the number of graph nodes matches the number of vertices.
    if graph.number_of_nodes() != n_vertices:
        raise ValueError(
            f"Graph has {graph.number_of_nodes()} nodes but 'values' has {n_vertices} vertices."
        )

    # Validate the dimensionality of values.
    if values.ndim not in (1, 2):
        raise ValueError("'values' should be either a 1D or 2D array.")
    
    # Build the adjacency list (each vertex's neighbors)
    adjacency_list = [[] for _ in range(n_vertices)]
    for u, v in graph.edges():
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
    
    # Prepare working buffers for iterative updates
    smoothed_map = values.copy()         # current smoothed values
    new_map = np.zeros_like(smoothed_map)  # buffer for updated values
    
    # Perform iterative smoothing
    for _ in range(iterations):
        if smoothed_map.ndim == 1:
            # 1D case: each vertex has a single value.
            for u in range(n_vertices):
                neighbors = adjacency_list[u]
                total = smoothed_map[u]  # Always include self.
                count = 1
                for v in neighbors:
                    total += smoothed_map[v]
                    count += 1
                new_map[u] = total / count
        else:
            # 2D case: each vertex has a vector of values (n_maps)
            for u in range(n_vertices):
                neighbors = adjacency_list[u]
                total = smoothed_map[u, :].copy()  # Include self.
                count = 1
                if neighbors:
                    total += smoothed_map[neighbors, :].sum(axis=0)
                    count += len(neighbors)
                new_map[u, :] = total / count
        
        # Swap buffers for the next iteration.
        smoothed_map, new_map = new_map, smoothed_map

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


