import os
from typing import Literal
import numpy as np
import networkx as nx
from datetime import datetime


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

def compute_gradient(graph, stat_map):
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


def compute_gradients(graph, similarity_matrix, skip=10):
    """Compute the gradients of the similarty matrix on a mesh surface.
    """
    gradients = np.zeros_like(similarity_matrix[0,:])
    for idx_map in range(0,similarity_matrix.shape[0], skip):
        stat_map = similarity_matrix[idx_map,:]
        gradient = np.zeros_like(stat_map)
        for vertex in graph.nodes:
            neighbors = list(graph.neighbors(vertex))
            if len(neighbors) == 0:
                continue
            # Compute the difference between the vertex and its neighbors
            differences = stat_map[neighbors] - stat_map[vertex]
            gradient[vertex] = np.sqrt(np.sum(differences ** 2))
        gradients += gradient
    return gradients




# Precise but costly method
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


def save_gradient_map(
        gradient_map,
        output_dir,
        hemisphere: Literal["lh", "rh"]
    ) -> None:
    """Save the gradient map into a .npy file
    Args:
        gradient_map (np.array): the gradient in order to the coords from the triangles surface
        output_dir (string): dir for grad output
        hemisphere (strinf): the hemisphere of the surface data
    """
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    path = output_dir + f"\{hemisphere}_gradient_map_{time}.npy"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, gradient_map)
    
def load_gradient_map(path):
    """Load the gradient map
    """
    gradient_map = np.load(path)
    return gradient_map