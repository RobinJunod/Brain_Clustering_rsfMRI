"""This file has sereval approches to calculate the 
gradient magnitude map of a 3D image. The gradient magnitude map is a 
3D image that contains the gradient magnitude of the input 3D image.

@myspace 2024-2025 EPFL
"""

import numpy as np


def compute_gradient_map(sim_matrix,
                         spatial_position,
                         roi_data_shape) -> np.ndarray:
    """Compute the gradient map from a similarity matrix using diffusion embedding.
    the gradient map is a 3D map of brain highligting place with similar connectivity.
    
    Args:
        sim_matrix (np.array): Computed from the similarity matrix
        spatial_position (tuple): Given by the function compute_similarity_matrix_in_ROI
        roi_data_shape (3D np.array): Shape of the ROI data

    Returns:
        np.ndarray: Gradient magnitude map
    """
    print('Computing gradient map...')
    # Unpack the spatial positions
    n_voxels = sim_matrix.shape[0]
    x_coords, y_coords, z_coords = spatial_position
    coord_to_index = {
        (x_coords[i], y_coords[i], z_coords[i]): i for i in range(len(x_coords))
    }
    # initialize the gradient magnitude array
    gradient_magnitude_map = np.zeros(roi_data_shape)
    # search for neighbours in the 3D space
    for i in range(n_voxels):
        x, y, z = x_coords[i], y_coords[i], z_coords[i]
        if sim_matrix[i].all() == 0:
            continue
        num_neighbours = 0
        rmse = 0
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            new_x, new_y, new_z = x + dx, y + dy, z + dz
            if (new_x, new_y, new_z) in coord_to_index:
                j = coord_to_index[(new_x, new_y, new_z)]
                rmse += np.linalg.norm(sim_matrix[i] - sim_matrix[j])
                num_neighbours += 1
        if num_neighbours > 0:
            gradient_magnitude_map[x, y, z] = rmse / num_neighbours
    return gradient_magnitude_map 

def compute_gradient_map_gaussian(sim_matrix,
                                  spatial_position,
                                  roi_data_shape,
                                  sigma=1.0,
                                  radius=2) -> np.ndarray:
    """Compute the gradient map from a similarity matrix using a Gaussian-weighted neighborhood.
    The gradient map is a 3D map highlighting places with similar connectivity.

    Args:
        sim_matrix (np.array): Computed from the similarity matrix.
        spatial_position (tuple): Given by the function compute_similarity_matrix_in_ROI.
        roi_data_shape (tuple): Shape of the ROI data (should be 3D).
        sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
        radius (int, optional): Radius of the neighborhood to consider. Defaults to 3.

    Returns:
        np.ndarray: Gradient magnitude map.
    """
    print('Computing gradient map with Gaussian-weighted neighborhood...')
    # Unpack the spatial positions
    n_voxels = sim_matrix.shape[0]
    x_coords, y_coords, z_coords = spatial_position
    coord_to_index = {
        (x_coords[i], y_coords[i], z_coords[i]): i for i in range(n_voxels)
    }
    # Initialize the gradient magnitude array
    gradient_magnitude_map = np.zeros(roi_data_shape)
    # Precompute the possible offsets and their Gaussian weights
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the center voxel
                distance_squared = dx**2 + dy**2 + dz**2
                weight = np.exp(-distance_squared / (2 * sigma**2))
                if weight > 1e-6:  # Ignore negligible weights
                    offsets.append((dx, dy, dz, weight))
    # Compute the gradient magnitude map
    for idx in range(n_voxels):
        x_i, y_i, z_i = x_coords[idx], y_coords[idx], z_coords[idx]
        if not np.any(sim_matrix[idx]):
            continue  # Skip if the similarity vector is all zeros
        weighted_rmse = 0.0
        total_weight = 0.0
        for dx, dy, dz, weight in offsets:
            x_j, y_j, z_j = x_i + dx, y_i + dy, z_i + dz
            # Check if the neighbor voxel is within the ROI
            if (x_j, y_j, z_j) in coord_to_index:
                jdx = coord_to_index[(x_j, y_j, z_j)]
                diff = np.linalg.norm(sim_matrix[idx] - sim_matrix[jdx])
                weighted_rmse += weight * diff
                total_weight += weight
        if total_weight > 0:
            gradient_magnitude_map[x_i, y_i, z_i] = weighted_rmse / total_weight
    
    return gradient_magnitude_map


def blur_gradient_map(gradient_magnitude_map: np.ndarray,
                      sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to the gradient magnitude map."""
    from scipy.ndimage import gaussian_filter
    print('Applying Gaussian blur to the gradient magnitude map...')
    blurred_map = gaussian_filter(gradient_magnitude_map, sigma=sigma)
    return blurred_map