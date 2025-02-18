#%%
import numpy as np
from nilearn.plotting import view_surf

def visualize_brain_surface(coords,
                            faces,
                            scalar_values,
                            title="Statistical map on surface",
                            cmap="viridis",
                            threshold=0):
    """
    Visualize scalar data on a triangular mesh using Nilearn.

    Args:
        coords : (N, 3) ndarray
            3D coordinates of each vertex.
        faces : (M, 3) ndarray
            Triangles as vertex indices.
        scalar_values : (N,) ndarray
            The scalar (e.g., gradient magnitude) for each vertex.
        title : str or None
            Title of the plot.
        cmap : str
            Name of the color map (e.g. "viridis", "coolwarm", etc.).
        threshold : float
            Threshold value for the scalar data.
    Returns:
        view : Nilearn plot
    """
    # Create a surface mesh
    surf_mesh = (coords, faces)
    high_threshold = np.percentile(scalar_values, threshold)

    # Visualize the scalar data on the surface
    view = view_surf(
        surf_mesh=surf_mesh,
        surf_map=scalar_values,
        cmap=cmap,
        bg_map=None,
        threshold=high_threshold,
        symmetric_cmap=False, 
        vmax=scalar_values.max(),
        vmin=scalar_values.min(),
        title=title,
    )

    return view

# Algorithm used to combine two surfaces (usually left and right hemispheres) into one mesh
def combine_surfaces(coords1, faces1, scalar1, coords2, faces2, scalar2):
    """
    Combine two surfaces (e.g., left and right hemispheres) into one mesh.
    
    Args:
        coords1, coords2: (N1, 3) and (N2, 3) arrays of coords.
        faces1, faces2: (M1, 3) and (M2, 3) arrays of faces.
        scalar1, scalar2: (N1,) and (N2,) arrays of scalar values.
    
    Returns:
        combined_coords: (N1+N2, 3) array.
        combined_faces: (M1+M2, 3) array (faces2 indices are shifted by N1).
        combined_scalar: (N1+N2,) array.
    """
    # Combine coords by stacking them vertically.
    combined_coords = np.vstack([coords1, coords2])
    
    # Adjust faces for the second surface by adding the number of coords in the first.
    offset = coords1.shape[0]
    faces2_adjusted = faces2 + offset
    combined_faces = np.vstack([faces1, faces2_adjusted])
    
    # Combine scalar values (make sure they are in the correct order).
    combined_scalar = np.concatenate([scalar1, scalar2])
    
    return combined_coords, combined_faces, combined_scalar

# Pyvista version for statistical map plotting on the surface
import pyvista as pv
import numpy as np
def visualize_brain_pyvista(coords, faces, values, cmap="viridis"):
    """
    Plot a surface using PyVista with interactive rotation for different views.
    
    Parameters:
    - coords (numpy.ndarray): Array of vertex coordinates, shape (N, 3).
    - faces (numpy.ndarray): Array of faces, shape (M, 3) or (M, 4) depending on triangle/quadrilateral meshes.
    - values (numpy.ndarray): Array of scalar values associated with each vertex, shape (N,).
    - cmap (str): Colormap for the surface visualization. Default is 'viridis'.
    """
    # Ensure faces are in the format PyVista expects
    if faces.shape[1] == 3:  # Triangular faces
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    elif faces.shape[1] == 4:  # Quadrilateral faces
        faces = np.hstack([np.full((faces.shape[0], 1), 4), faces])
    else:
        raise ValueError("Faces should have either 3 or 4 vertices per face.")

    # Create the PyVista mesh
    mesh = pv.PolyData(coords, faces)
    mesh.point_data["values"] = values  # Add scalar values to the mesh

    # Set up the PyVista plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars="values", cmap=cmap, show_scalar_bar=True)
    plotter.show_axes()  # Show axes for reference
    plotter.view_isometric()  # Set an initial isometric view

    # Display the interactive plot
    plotter.show()


# Visualization to plot two different surfaces
import matplotlib.pyplot as plt
from nilearn import plotting

def visualize_surfaces_side_by_side(
    coords1, faces1, values1,
    coords2, faces2, values2
):
    fig, axes = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"}, figsize=(10, 5))
    
    # First surface
    plotting.plot_surf(
        surf_mesh=(coords1, faces1),
        surf_map=values1,
        cmap='viridis',
        colorbar=True,
        axes=axes[0],
        title="Surface 1"
    )
    
    # Second surface
    plotting.plot_surf(
        surf_mesh=(coords2, faces2),
        surf_map=values2,
        cmap='viridis',
        colorbar=True,
        axes=axes[1],
        title="Surface 2"
    )
    
    plt.tight_layout()
    plt.show()


