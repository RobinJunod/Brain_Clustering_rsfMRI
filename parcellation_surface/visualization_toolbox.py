#%%
import numpy as np
from nilearn.plotting import view_surf

def visualize_brain_surface(vertices, faces, scalar_values, cmap="viridis"):
    """
    Visualize scalar data on a triangular mesh using Nilearn.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        3D coordinates of each vertex.
    faces : (M, 3) ndarray
        Triangles as vertex indices.
    scalar_values : (N,) ndarray
        The scalar (e.g., gradient magnitude) for each vertex.
    cmap : str
        Name of the color map (e.g. "viridis", "coolwarm", etc.).
    """

    # Create a surface mesh
    surf_mesh = (vertices, faces)

    # Visualize the scalar data on the surface
    view = view_surf(
        surf_mesh=surf_mesh,
        surf_map=scalar_values,
        cmap=cmap,
        bg_map=None,
        title="Gradient Magnitude",
    )

    return view




##########################
import numpy as np

def build_adjacency_list(faces, n_vertices):
    """
    Build a list of neighbors for each vertex.
    """
    neighbors = [[] for _ in range(n_vertices)]
    for tri in faces:
        i, j, k = tri
        neighbors[i].append(j)
        neighbors[i].append(k)
        neighbors[j].append(i)
        neighbors[j].append(k)
        neighbors[k].append(i)
        neighbors[k].append(j)
    # Remove duplicates
    for v in range(n_vertices):
        neighbors[v] = list(set(neighbors[v]))
    return neighbors

def compute_vertex_normals(coords, faces):
    """
    Approximate vertex normals by averaging the face normals of 
    all faces incident on each vertex.
    coords: Nx3
    faces: Mx3
    Returns: Nx3 array of vertex normals (not normalized).
    """
    n_vertices = coords.shape[0]
    # Face normals
    v1 = coords[faces[:,1]] - coords[faces[:,0]]
    v2 = coords[faces[:,2]] - coords[faces[:,0]]
    face_normals = np.cross(v1, v2)  # Mx3
    
    # Accumulate face normals to vertices
    vertex_normals = np.zeros((n_vertices, 3), dtype=np.float64)
    for f_idx, (i, j, k) in enumerate(faces):
        fn = face_normals[f_idx]
        vertex_normals[i] += fn
        vertex_normals[j] += fn
        vertex_normals[k] += fn
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-9
    vertex_normals = vertex_normals / norms
    return vertex_normals

def inflate_like_balloon(coords, faces, n_iter=100, step_normal=0.1, step_smooth=0.1):
    """
    Inflate a mesh 'like a balloon':
      - push vertices outward along their normals
      - apply mild smoothing to reduce local folds
    
    coords: Nx3 array of vertex positions
    faces: Mx3 array of triangle indices (int)
    n_iter: number of inflation steps
    step_normal: how far to push along the normal each iteration
    step_smooth: how strongly to smooth (Laplacian) each iteration

    Returns: Nx3 array (inflated coordinates)
    """
    new_coords = coords.copy()
    n_vertices = new_coords.shape[0]
    neighbors = build_adjacency_list(faces, n_vertices)

    for iteration in range(n_iter):
        # 1) Compute vertex normals
        vertex_normals = compute_vertex_normals(new_coords, faces)
        
        # 2) "Balloon" step: push outward along normals
        #    If you want to preserve volume, you might rescale or clamp later.
        new_coords += step_normal * vertex_normals
        
        # 3) Mild Laplacian smoothing
        #    For each vertex, move it slightly toward average of neighbors
        smoothed_coords = new_coords.copy()
        for v in range(n_vertices):
            neigh = neighbors[v]
            if not neigh: 
                continue
            avg_neighbor = np.mean(new_coords[neigh], axis=0)
            smoothed_coords[v] = (
                (1.0 - step_smooth) * new_coords[v] 
                + step_smooth * avg_neighbor
            )
        
        new_coords = smoothed_coords
    
    return new_coords