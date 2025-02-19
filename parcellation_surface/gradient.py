import os
from typing import Literal
import nibabel as nib
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


# Most efficient method for large matrix gradient computation
def compute_gradients(graph, 
                      similarity_matrix):
    """
    Compute a local gradient magnitude at each vertex for each column of a 'similarity_matrix',
    using an edge-based vectorized approach. This give all N vertices and M maps. Fast and efficient.
    
    Args:
        graph (networkx.Graph) : Undirected graph representing the mesh connectivity (one node per vertex).
            Nodes should be laeled 0..(N-1).
        similarity_matrix (np.ndarray) (N vertices, M maps) : 'N' vertices and 'M' maps/columns. Column m is the scalar field for map m.
    Returns
        gradients (np.ndarray) : (N vertices, M maps) : gradients[v,m] = sqrt( sum_{u in neighbors(v)} (vals[v,m] - vals[u,m])^2 )
    """
    # Convert input to a NumPy array of shape (N, M)
    similarity_matrix = np.asarray(similarity_matrix, dtype=np.float16)
    N, M = similarity_matrix.shape

    # We will accumulate sum of squared differences for each vertex v, for each map m
    squared_sums = np.zeros((N, M), dtype=np.float16)

    # Loop once over all edges
    for v, u in graph.edges():
        # For each edge (v,u), compute the difference across all maps at once
        diff = similarity_matrix[v, :] - similarity_matrix[u, :]  # shape (M,)
        diff_sq = diff * diff  # elementwise square

        # Add the squared difference to both endpoints v and u
        squared_sums[v, :] += diff_sq
        squared_sums[u, :] += diff_sq

    # Now take the sqrt => final gradient magnitude at each vertex, for each map
    gradients = np.sqrt(squared_sums)

    return gradients

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Other old methods for gradient computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

def compute_gradient_average(graph, 
                            similarity_matrix, 
                            skip=10):
    """Compute the gradients of the similarty matrix on a mesh surface.
    Args:
        graph (networkx.Graph): the mesh graph
        similarity_matrix (np.array): the similarity matrix with maps stored in columns [n_vertices, n_maps]
        skip (int): the number of vertices to skip
    Returns:
        gradients (np.array): the gradients of the similarity matrix
    """
    gradients = np.zeros_like(similarity_matrix[:,0])
    for idx_map in range(0,similarity_matrix.shape[0], skip):
        stat_map = similarity_matrix[:,idx_map]
        gradient = np.zeros_like(stat_map)
        for vertex in graph.nodes:
            neighbors = list(graph.neighbors(vertex))
            if len(neighbors) == 0:
                continue
            # Compute the difference between the vertex and its neighbors (return a vector with all neighbors)
            differences = stat_map[neighbors] - stat_map[vertex]
            gradient[vertex] = np.sqrt(np.sum(differences ** 2))
        gradients += gradient
    return gradients




# Precise but costly method
def compute_gradient_magnitudes(faces, coords, values): 
    """
    Compute per-vertex gradient magnitudes of a scalar field on a triangular mesh.

    Args:
        faces (np.ndarray): Triangle vertex indices, shape (n_faces, 3).
        coords (np.ndarray): Vertex coordinates, shape (n_vertices, 3).
        values (np.ndarray): Scalar field values at each vertex, shape (n_vertices,).
        
    Returns:
        grad_magnitudes (np.ndarray): Gradient magnitudes at each vertex, shape (n_vertices,).
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


def save_gradient_mgh(gradient_map,
                      output_dir,
                      hemisphere: Literal["lh", "rh"],
                      name = "gradient"):
    """Save the gradient map into a .mgh file
    Args:
        gradient_map (np.array): the gradient in order to the coords from the triangles surface
        output_dir (string): dir for grad output
        hemisphere (strinf): the hemisphere of the surface data
        name (string): name of the gradient file
    """
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    # Reshape the data to match the FreeSurfer .mgh format expectations
    gradient_reshaped = gradient_map.reshape((len(gradient_map), 1, 1)).astype(np.float32)
    # Create an identity affine (often used for surface data).
    affine = np.eye(4)
    # Construct the MGH image.
    mgh_img = nib.freesurfer.mghformat.MGHImage(gradient_reshaped, affine)
    # Save the MGH file to disk.
    path = output_dir + f"\{name}_{hemisphere}_{time}.mgh"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(mgh_img, path)


def load_gradient_mgh(mgh_file_path):
    # Load the MGH image
    mgh_image = nib.load(mgh_file_path)
    # Extract data as float32 (optional) and squeeze to remove single-dimensional axes
    data = mgh_image.get_fdata().squeeze()
    return data