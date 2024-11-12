"""This file has sereval approches to calculate the 
gradient magnitude map of a 3D image. The gradient magnitude map is a 
3D image that contains the gradient magnitude of the input 3D image.

THE BASELINE METHOD IS THE WIG 2014 METHOD 
'An approach for parcellating human cortical areas using restingstate correlations', Wig et al. 2014

@myspace 2024-2025 EPFL
contact : robin.junod@epfl.ch
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

    
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


# GRADIENT MAGNITUDE MAP WIG 2014 METHOD
def blur_gradient_map(gradient_magnitude_map: np.ndarray,
                      sigma: float = 0.5) -> np.ndarray:
    """Apply Gaussian blur to the gradient magnitude map."""
    from scipy.ndimage import gaussian_filter
    print('Applying Gaussian blur to the gradient magnitude map...')
    blurred_map = gaussian_filter(gradient_magnitude_map, sigma=sigma)
    return blurred_map


    
    
#CREATE A CUSTOM 3D DATASET FOR TESTING
def create_3d_test_data(function, shape= (50, 50, 50) , center=(25, 25, 25)  , radius= 15):
    """
    Creates a 3D array with a spherical ROI, where values inside the ROI are defined by a function
    and outside the ROI are zero.

    Parameters:
    - shape (tuple): The shape of the 3D array, e.g., (size_x, size_y, size_z).
    - center (tuple): The (x, y, z) coordinates for the center of the spherical ROI.
    - radius (float): Radius of the spherical ROI.
    - function (callable): Function to generate values inside the ROI, based on distance from center.

    Returns:
    - array (np.array): 3D array with values in a spherical ROI and zeros outside.
    """
    # Initialize a 3D array with zeros
    array = np.zeros(shape, dtype=float)
    
    # Create a grid of indices
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = np.arange(shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate distances from the center
    distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
    
    # Define the spherical ROI based on the radius
    roi_mask = distances <= radius
    
    # Apply the function within the ROI and set values outside ROI to zero
    array[roi_mask] = function(distances[roi_mask])
    
    return array, roi_mask
# Define a sinusoidal function for values inside the ROI
def sinusoidal_function(distance, radius=15):
    wavelength = radius / 4   # Adjust the wavelength to control oscillation frequency
    return np.sin(2 * np.pi * distance / wavelength) + 10

# Generate the 3D array with sinusoidal values inside the spherical ROI
# volume, roi_mask = create_3d_test_data(sinusoidal_function)

# CREATE A 3D VOLUME FROM THE FLATTENED SIMILARITY MAP
def flat2volum(flatten_array : np.array, # 1xn
               voxel_position : np.array): # 3xn
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

# COMPUTE THE SPATIAL MAGNITUDE GRADIENT INSIDE OF A ROI
def compute_gradient_inside_ROI(volume, 
                                roi_mask):
    """
    Compute the gradient magnitude within a specified ROI in a 3D volume.

    Parameters:
    - volume (np.ndarray): 3D array representing the volume.
    - roi_mask (np.ndarray): 3D boolean array where True indicates the ROI.

    Returns:
    - gradient_magnitude (np.ndarray): 3D array of the same shape as volume containing the gradient magnitudes within the ROI.
    """
    from scipy.ndimage import sobel, distance_transform_edt
    # Ensure roi_mask is boolean
    roi_mask = roi_mask.astype(bool)
    # Step 1: Replace values outside the ROI with the nearest ROI value to avoid artificial gradients
    # Compute the distance transform and obtain the indices of the nearest ROI voxel
    distance, indices = distance_transform_edt(~roi_mask, return_indices=True)
    padded_volume = volume[indices[0], indices[1], indices[2]]
    # Step 2: Compute gradients using the Sobel operator
    gx = sobel(padded_volume, axis=0, mode='constant')
    gy = sobel(padded_volume, axis=1, mode='constant')
    gz = sobel(padded_volume, axis=2, mode='constant')
    # Step 3: Compute the gradient magnitude
    gradient_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
    # Step 4: Mask the gradient magnitude to include only the ROI
    gradient_magnitude = np.where(roi_mask, gradient_magnitude, 0)
    
    gx = np.where(roi_mask, gx, 0)
    gy = np.where(roi_mask, gy, 0)
    gz = np.where(roi_mask, gz, 0)
    
    return gradient_magnitude, (gx, gy, gz)


# APPLY GAUSSIAN BLURRING
def gaussian_blurring(volume_scalar : np.array,
                      sigma : float = 1):
    """Takes a 3D numpy array and apply a gaussian kernel to blur it

    Args:
        volume_scalar (np.array): 3D np.array of with each idx representing a 
                                    discret postion of the voxel.
        sigma (float): variance of the 3D gaussian used
    """
    # Apply Gaussian filter to blur the volume
    blurred_volume = gaussian_filter(volume_scalar, sigma=sigma)
    return blurred_volume

# APPLY THE NON MAXIMA ALGORITHM FOR A BETTER EDGE DETECTION
def non_maxima_suppression_3d(gradient_magnitude, gx, gy, gz, roi_mask):
    """
    Perform 3D Non-Maxima Suppression on the gradient magnitude within a ROI.

    Parameters:
    - gradient_magnitude (np.ndarray): 3D array of gradient magnitudes.
    - gx, gy, gz (np.ndarray): 3D arrays of gradient components along x, y, z axes.
    - roi_mask (np.ndarray): 3D boolean array where True indicates the ROI.

    Returns:
    - nms (np.ndarray): 3D array after non-maxima suppression.
    """
    from scipy.ndimage import map_coordinates
    # Ensure inputs are float for precision
    gradient_magnitude = gradient_magnitude.astype(np.float32)
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    gz = gz.astype(np.float32)
    roi_mask = roi_mask.astype(bool)
    
    # Initialize the output array
    nms = np.zeros_like(gradient_magnitude)
    
    # Get the indices of all voxels within the ROI
    indices = np.array(np.nonzero(roi_mask)).T  # Shape: (num_voxels, 3)
    
    for idx in indices:
        x, y, z = idx
        # Get the gradient vector at this voxel
        g_x = gx[x, y, z]
        g_y = gy[x, y, z]
        g_z = gz[x, y, z]
        
        # Normalize the gradient vector
        norm = np.sqrt(g_x**2 + g_y**2 + g_z**2)
        if norm == 0:
            continue  # Cannot determine direction; skip
        g_x /= norm
        g_y /= norm
        g_z /= norm
        
        # Determine the two neighboring voxel positions
        neighbor1 = [x + g_x, y + g_y, z + g_z]
        neighbor2 = [x - g_x, y - g_y, z - g_z]
        
        # Check boundaries
        if (0 <= neighbor1[0] < gradient_magnitude.shape[0] - 1 and
            0 <= neighbor1[1] < gradient_magnitude.shape[1] - 1 and
            0 <= neighbor1[2] < gradient_magnitude.shape[2] - 1 and
            0 <= neighbor2[0] < gradient_magnitude.shape[0] - 1 and
            0 <= neighbor2[1] < gradient_magnitude.shape[1] - 1 and
            0 <= neighbor2[2] < gradient_magnitude.shape[2] - 1):
            
            # Interpolate gradient magnitudes at the neighboring positions
            gm1 = map_coordinates(gradient_magnitude, 
                                  [[neighbor1[0]], [neighbor1[1]], [neighbor1[2]]], 
                                  order=1, mode='nearest')[0]
            gm2 = map_coordinates(gradient_magnitude, 
                                  [[neighbor2[0]], [neighbor2[1]], [neighbor2[2]]], 
                                  order=1, mode='nearest')[0]
            
            # Suppress if not a local maximum
            if gradient_magnitude[x, y, z] >= gm1 and gradient_magnitude[x, y, z] >= gm2:
                nms[x, y, z] = gradient_magnitude[x, y, z]
    
    return nms

def pipeline_wig2014(sim_mtrx,
                     spatial_position,
                     volumne_shape : tuple=(91,109,91)):
    """This compute for a single subject the gradient magnitude map.
    This map highlights the zone of the brain with a similar activity.

    Args:
        sim_mtrx (np.array): The similarity matrix. each columns or row is a similarity map
        spatial_position (np.array): A 3xn array of the spatial position of the voxel. 
                                    Each column is a voxel position.
    
    Returns:
        edge_maps_mean (np.array): The mean edge map of the similarity maps.
    """
    
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
    # readjust the spatial position
    position_adjusted = np.array([x_adjusted, 
                                  y_adjusted,
                                  z_adjusted])
    # Initialize a mask for the region of interest
    roi_adjusted = np.zeros((size_x, size_y, size_z))
    roi_adjusted[x_adjusted, y_adjusted, z_adjusted] = 1
    # Initialize the edge map
    edge_maps_mean = np.zeros_like(roi_adjusted)
    # loop across each columns in sim matrix (coressponding to a sim map)
    for sim_map_id in range(sim_mtrx.shape[0]):
        # Extract a similarty map
        sim_map_flat = sim_mtrx[:,sim_map_id]
        # transform the sim map in 3D
        sim_map_3d = flat2volum(sim_map_flat, position_adjusted)
        # TODO : compare without blurring the sim map
        # sim_map_3d_blurr = gaussian_blurring(sim_map_3d, sigma=1)
        # Compute the gradient magnitude map
        grad_map, (gx,gy,gz) = compute_gradient_inside_ROI(sim_map_3d, roi_adjusted)
        # Detect edges form sim_map
        edge_map = non_maxima_suppression_3d(grad_map, gx, gy, gz , roi_adjusted)
        # Compute the mean edge map
        edge_maps_mean = edge_maps_mean + edge_map/sim_mtrx.shape[0]
    
    
    # Place the edge map back in the original volume
    final_edge_map = np.zeros(volumne_shape)
    final_edge_map[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = edge_maps_mean
    return final_edge_map

#%%
if __name__ == '__main__':
    import os
    import scipy
    from datetime import datetime
    import nibabel as nib
    # Load a similartiy matrix
    outdir_sim_mtrx = r'G:/DATA_min_preproc/dataset_study1/S02/outputs/sim_mtrx'
    out_base_name = 'similarity_matrix_S02_20241106_180152'
    file_path = os.path.join(outdir_sim_mtrx, out_base_name )
    loaded_data = scipy.io.loadmat(file_path)
    sim_mtrx = loaded_data['S']
    spatial_position = loaded_data['spatial_position'] 
    #%%
    edge_map = pipeline_wig2014(sim_mtrx, 
                                spatial_position,
                                volumne_shape=(91,109,91))
    
    # Save the edge map as a NIfTI file
    from toolbox_parcellation import extract_nii_files
    outdir_grad_map = r'G:/DATA_min_preproc/dataset_study1/S02/outputs/edge_map'
    out_base_name = f'mean_edge_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # _, _, _, original_affine = extract_nii_files(fmri_path, roi_mask_path, brain_mask_path, output_dir)

    # Ensure the output directory exists
    os.makedirs(outdir_grad_map, exist_ok=True)
    nii_img = nib.Nifti1Image(edge_map, affine=original_affine)
    nib.save(nii_img, os.path.join(outdir_grad_map, out_base_name + '.nii'))# %%

# %%
