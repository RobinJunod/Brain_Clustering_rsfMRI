# This file contians methods for the computation of the gradient
# of a similarity map on a mesh surface. The gradient is computed
# using the finite difference method.


import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

def build_adjacency_matrix(faces, n_vertices):
    """
    Build a sparse adjacency matrix from mesh faces.
    
    faces: ndarray of shape (n_faces, 3)
    n_vertices: int, total number of vertices
    
    Returns:
        adjacency_matrix: scipy.sparse.csr_matrix of shape (n_vertices, n_vertices)
    """
    # Extract edges
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    
    # Remove duplicate edges
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    
    # Create adjacency matrix
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)
    adjacency_matrix = csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
    
    # Since the graph is undirected, add the transpose
    adjacency_matrix = adjacency_matrix + adjacency_matrix.T
    
    return adjacency_matrix


def compute_gradients(similarity_map, adjacency_matrix):
    """
    Compute the gradient magnitude at each vertex based on similarity map using adjacency matrix.
    
    similarity_map: ndarray of shape (n_vertices,)
    adjacency_matrix: scipy.sparse.csr_matrix of shape (n_vertices, n_vertices)
    
    Returns:
        gradients: ndarray of shape (n_vertices,)
    """
    # Compute differences with neighbors
    differences = adjacency_matrix.dot(similarity_map.reshape(-1, 1)) - similarity_map.reshape(-1, 1) * adjacency_matrix.dot(np.ones((adjacency_matrix.shape[1], 1)))
    gradients = np.sqrt(np.sum(differences ** 2, axis=1)).flatten()
    return gradients


def non_maxima_suppression(gradient_map, adjacency_matrix, threshold=0):
    """
    Apply non-maxima suppression to identify edge vertices.
    
    gradient_map: ndarray of shape (n_vertices,)
    adjacency_matrix: scipy.sparse.csr_matrix of shape (n_vertices, n_vertices)
    threshold: float, minimum gradient magnitude to be considered as edge
    
    Returns:
        edge_map: ndarray of shape (n_vertices,), boolean
    """
    # For each vertex, get the maximum gradient in its neighborhood
    max_neighbor_gradients = adjacency_matrix.multiply(gradient_map).max(axis=1).A1
    # A vertex is an edge if its gradient is greater than all its neighbors and above the threshold
    edge_map = (gradient_map > max_neighbor_gradients) & (gradient_map > threshold)
    return edge_map


def first_method_gradient_map(coords, faces, values):
    """
    Compute the gradient of a scalar field on a mesh using the first method.
    
    coords: ndarray of shape (n_vertices, 3)
    faces: ndarray of shape (n_faces, 3)
    values: ndarray of shape (n_vertices,)
    
    Returns:
        gradients: ndarray of shape (n_vertices,)
    """
    # Build adjacency matrix
    n_vertices = len(coords)
    adjacency_matrix = build_adjacency_matrix(faces, n_vertices)
    
    # Compute gradients
    gradients = compute_gradients(values, adjacency_matrix)
    
    # Detect the edges
    edge_map = non_maxima_suppression(gradients, adjacency_matrix)
    
    return edge_map



