#%%
import numpy as np
import pyvista as pv

def visualize_brain_scalar(vertices, faces, scalar_values, cmap="viridis"):
    """
    Visualize scalar data on a triangular mesh using PyVista.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        3D coordinates of each vertex.
    faces : (M, 3) ndarray or (M*4,) ndarray
        Triangles as vertex indices. PyVista typically expects a "faces array"
        where each triangle is specified as [3, v0, v1, v2] for each face.
        If you only have the (M, 3) array of indices, you can convert it.
    scalar_values : (N,) ndarray
        The scalar (e.g., gradient magnitude) for each vertex.
    cmap : str
        Name of the color map (e.g. "viridis", "coolwarm", etc.).
    """

    # If your faces are shape (M, 3), then you must prepend each row with '3'
    # to create PyVista's "faces" format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    if faces.ndim == 2 and faces.shape[1] == 3:
        num_faces = faces.shape[0]
        faces_pv = np.column_stack([
            3 * np.ones(num_faces, dtype=np.int64),
            faces
        ]).ravel()
    else:
        faces_pv = faces  # If already in PyVista format

    # Create the PolyData object
    mesh = pv.PolyData(vertices, faces_pv)

    # Add the scalar data (PyVista calls them "point_data")
    mesh.point_data["my_scalar"] = scalar_values

    # Plot with PyVista
    plotter = pv.Plotter()
    actor = plotter.add_mesh(
        mesh,
        scalars="my_scalar",
        cmap=cmap,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Gradient Magnitude"},
    )
    plotter.show()
    
    
    
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
        vmax=np.percentile(scalar_values, 95),
        bg_map=None,
        title="Gradient Magnitude",
    )

    return view
