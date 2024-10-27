"""This file has sereval approches to calculate the 
gradient magnitude map of a 3D image. The gradient magnitude map is a 
3D image that contains the gradient magnitude of the input 3D image.

@myspace 2024-2025 EPFL
"""

#%%
import numpy as np

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, sobel
import scipy.ndimage as ndi


def gaussian_blurring(volume_scalar : np.array,
                      sigma : float = 2):
    """Takes a 3D numpy array and apply a gaussian kernel to blur it

    Args:
        volume_scalar (np.array): 3D np.array of with each idx representing a 
                                    discret postion of the voxel.
        sigma (float): variance of the 3D gaussian used
    """
    # Apply Gaussian filter to blur the volume
    blurred_volume = gaussian_filter(volume_scalar, sigma=sigma)
    return blurred_volume


def flat2volum(flatten_array : np.array, # 1xn
               voxel_position : np.array,# 3xn
               ):
    """From an array of scalar values and a matrix of (integer) the postion of these scalar.
    Reconstruct a 3D np.array with the positon of the voxel as index.

    Args:
        flatten_array (np.array): _description_
        voxel_position (np.array): _description_
    """
    # Initialize the volume with zeros
    max_indices = voxel_position.max(axis=1)
    volume_shape = tuple(max_indices + 1)
    volume = np.zeros(volume_shape)
    # Extract voxel indices
    x_indices, y_indices, z_indices = voxel_position
    # Place scalar values at the corresponding positions
    volume[x_indices, y_indices, z_indices] = flatten_array
    return volume

def gradient_magnitude(volume: np.array):
    """An algorithm that computes the first spatial derivate. Which gives a gradient.
    The magnitude of this gradient is computed for each voxels

    Args:
        volume (np.array): a 3D numpy array with each values representing a voxel

    Returns:
        np.array: gradient magnitude map
    """
    # Compute gradients along each axis
    grad_x, grad_y, grad_z = np.gradient(volume)
    
    # Compute gradient magnitude at each voxel
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    return gradient_magnitude


def gradient_map_wig2014(sim_mtrx,
                         spatial_position):
    """Compute the average edge map from similarity maps.
    
    Args:
        sim_mtrx (numpy.ndarray): n x n similarity matrix.
        spatial_position (numpy.ndarray): 3 x n array of spatial positions.
    
    Returns:
        numpy.ndarray: Averaged edge map.
    """
    # (1 subject) Each column of the sim_mtrx represent a similartiy map
    # For each similarity map:
    # apply gaussian blurring with sigma=2.55
    # compute the gradient of each similarty map
    # edge detection of the similarity grad map
    # make average of each edge map

    # Extract voxel indices (get a cube of the ROI)
    x_indices = spatial_position[0]
    y_indices = spatial_position[1]
    z_indices = spatial_position[2]
    # Determine the array size
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    z_min, z_max = z_indices.min(), z_indices.max()
    size_x = x_max - x_min + 1
    size_y = y_max - y_min + 1
    size_z = z_max - z_min + 1
    # Adjust indices to start from zero
    x_adjusted = x_indices - x_min
    y_adjusted = y_indices - y_min
    z_adjusted = z_indices - z_min
    
    position_adjusted = np.array([x_adjusted, 
                                  y_adjusted,
                                  z_adjusted])
    

    
    edge_maps = []
    
    # loop across each columns in sim matrix (coressponding to a sim map)
    for sim_map_id in range(sim_mtrx.shape[0]):
        # Your scalar values array of length n
        sim_map_flat = sim_mtrx[:,sim_map_id] # Replace with your data
        # Create sim map in 3d
        sim_map_3d = flat2volum(sim_map_flat, position_adjusted)
        # Compute the gradient magnitude map
        grad_map = gradient_magnitude(sim_map_3d)
        # Blurring the sim map
        grad_map_blur = gaussian_blurring(grad_map)
        # Detect edges form sim_map
        


    return edge_maps
    
    
    
# GRADIENT MAGNITUDE MAP CUSTOM METHOD
def custom_gradient_map(sim_matrix,
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

def custom_gradient_map_gaussian(sim_matrix,
                                  spatial_position,
                                  roi_data_shape,
                                  sigma=3.0,
                                  radius=3) -> np.ndarray:
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


#%%
if __name__=='__main__':
    #The metric-gradient-all function in Caret 5.65 computes the magnitude of the gradient at each verte
    # create test dataset 
    # brain mask
    # brain roi
    # create 
    from toolbox_parcellation import extract_4Ddata_from_nii, extract_3Ddata_from_nii
    path_roi = r'C:\Users\Robin\Documents\1_EPFL\PDMe\data\ROI\ROI_postcentral.nii'
    path_mask =  r'C:\Users\Robin\Documents\1_EPFL\PDMe\data\ROI\MASK_wholebrain.nii'
    path_fmri =  r'C:\Users\Robin\Documents\1_EPFL\PDMe\data\HCP\rfMRI_REST1_LR.nii'
    # extract data and affine transformation matrix
    fmri_data, _ = extract_4Ddata_from_nii(path_fmri)
    roi_data, original_affine = extract_3Ddata_from_nii(path_roi)
    mask_data, _ = extract_3Ddata_from_nii(path_mask)
    
# %%
import numpy as np
import matplotlib.pyplot as plt

def visualize_slices(volume, axis=2, slice_indices=None, cmap='gray'):
    """
    Visualize slices of a 3D NumPy array.

    Args:
        volume (np.array): The 3D NumPy array to visualize.
        axis (int): The axis along which to take slices (0 for x, 1 for y, 2 for z).
        slice_indices (int or list of int, optional): Indices of the slices to visualize. 
                                                      If None, the middle slice is displayed.
        cmap (str): Colormap to use for displaying the slices.

    """
    if slice_indices is None:
        # If no indices are provided, display the middle slice
        idx = volume.shape[axis] // 2
        slice_indices = [idx]
    elif isinstance(slice_indices, int):
        slice_indices = [slice_indices]
    elif not isinstance(slice_indices, (list, tuple)):
        raise TypeError("slice_indices must be an int, list, or None.")

    num_slices = len(slice_indices)
    fig, axes = plt.subplots(1, num_slices, figsize=(5 * num_slices, 5))
    
    # If only one subplot, put axes in a list for consistency
    if num_slices == 1:
        axes = [axes]
    
    for ax, idx in zip(axes, slice_indices):
        if axis == 0:
            # Slice along the x-axis
            slice_img = volume[idx, :, :]
            axis_name = 'X'
        elif axis == 1:
            # Slice along the y-axis
            slice_img = volume[:, idx, :]
            axis_name = 'Y'
        elif axis == 2:
            # Slice along the z-axis
            slice_img = volume[:, :, idx]
            axis_name = 'Z'
        else:
            raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z).")
        
        ax.imshow(slice_img, cmap=cmap)
        ax.set_title(f'Slice along {axis_name}-axis at index {idx}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()