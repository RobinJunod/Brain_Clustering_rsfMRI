"""This file has an implementation of the watershed by flooding algorithm for 3D dataset. 
The watershed algorithm is a classical algorithm used for image segmentation.

This algorithm needs a gradient magnitude map.

@myspace 2024-2025 EPFL
"""
#%%
import numpy as np
import heapq
from scipy import ndimage
import matplotlib.pyplot as plt
from toolbox_parcellation import visualize_slices


def find_seeds_basic(gradient_magnitude_map: np.ndarray) -> np.ndarray:
    """Find the seeds for the watershed algorithm.
    Args:
        gradient_magnitude_map (3D np.ndarray): Gradient magnitude map
    Returns:
        np.ndarray: Seeds for the watershed algorithm
    """
    spatial_position = np.where(gradient_magnitude_map > 0)
    n_voxels = len(spatial_position[0])
    
    
    x_coords, y_coords, z_coords = spatial_position
    coord_to_index = {
        (x_coords[i], y_coords[i], z_coords[i]): i for i in range(len(x_coords))
    }
    # intialize the seeds map (0 for no seed, 1 for seed)
    seeds_maps = np.zeros(gradient_magnitude_map.shape)
    # search for the seeds in the 3D space
    for i in range(n_voxels):
        x, y, z = x_coords[i], y_coords[i], z_coords[i]
        n_higher_grad_neighbours = 0
        # Generate all neighbors in the 3x3x3 cube, including the center voxel itself
        neighbors = [(x + dx, y + dy, z + dz)
                    for dx in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    for dz in [-1, 0, 1]]
        neighbors.remove((x, y, z))
        for new_x, new_y, new_z in neighbors:
            # check if the neighbour has smaller value
            if (new_x, new_y, new_z) in coord_to_index:
                if gradient_magnitude_map[new_x, new_y, new_z] < gradient_magnitude_map[x, y, z]:
                    break
            n_higher_grad_neighbours += 1
        if n_higher_grad_neighbours > 25:
            seeds_maps[x, y, z] = 1 # seed 
            
    return seeds_maps

def find_seeds(gradient_magnitude_map: np.ndarray,
               roi: np.ndarray,
               min_distance: float = 2.0) -> np.ndarray:
    """
    Find the seeds for the watershed algorithm, ensuring that seeds are not too close to each other.
    When seeds are closer than `min_distance`, they are merged and replaced by a single seed at the center of mass.

    Args:
        gradient_magnitude_map (3D np.ndarray): Gradient magnitude map
        roi (3D np.ndarray): Region of interest mask
        min_distance (float): Minimum distance between seeds

    Returns:
        np.ndarray: Seeds for the watershed algorithm
    """
    from scipy.spatial.distance import cdist
    # Identify voxels within the ROI
    spatial_position = np.array(np.where(roi > 0)).T
    n_voxels = len(spatial_position)

    # Initialize the seeds map (0 for no seed, 1 for seed)
    seeds_map = np.zeros(gradient_magnitude_map.shape, dtype=np.int32)

    # Identify local minima within the ROI
    local_minima_coords = []
    for idx in range(n_voxels):
        x, y, z = spatial_position[idx]
        is_minimum = True

        # Generate all neighbors in the 3x3x3 cube excluding the center voxel itself
        for dx in [-1, 0, 1]:
            xi = x + dx
            if xi < 0 or xi >= gradient_magnitude_map.shape[0]:
                continue
            for dy in [-1, 0, 1]:
                yi = y + dy
                if yi < 0 or yi >= gradient_magnitude_map.shape[1]:
                    continue
                for dz in [-1, 0, 1]:
                    zi = z + dz
                    if zi < 0 or zi >= gradient_magnitude_map.shape[2]:
                        continue
                    if (dx == 0 and dy == 0 and dz == 0):
                        continue
                    if roi[xi, yi, zi] > 0:
                        if gradient_magnitude_map[xi, yi, zi] < gradient_magnitude_map[x, y, z]:
                            is_minimum = False
                            break
                if not is_minimum:
                    break
            if not is_minimum:
                break

        if is_minimum:
            local_minima_coords.append((x, y, z))

    # Convert list to numpy array for efficient computation
    local_minima_coords = np.array(local_minima_coords)

    if len(local_minima_coords) == 0:
        return seeds_map  # No seeds found

    # Compute pairwise distances between minima
    distances = cdist(local_minima_coords, local_minima_coords)

    # Use a clustering approach to merge seeds that are too close
    # Create a connectivity graph where edges exist between seeds closer than min_distance
    adjacency_matrix = distances < min_distance
    np.fill_diagonal(adjacency_matrix, 0)  # Remove self-connections

    # Label connected components in the adjacency matrix
    # First, we need to create an undirected graph from the adjacency matrix
    from scipy.sparse import csgraph
    n_seeds = local_minima_coords.shape[0]
    graph = csgraph.csgraph_from_dense(adjacency_matrix, null_value=0)
    n_components, labels = csgraph.connected_components(csgraph=graph, directed=False)

    # For each cluster of seeds, compute the center of mass and place a single seed there
    for cluster_label in range(n_components):
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_coords = local_minima_coords[cluster_indices]
        # Compute the center of mass (mean position)
        com = np.mean(cluster_coords, axis=0).astype(int)
        # Ensure the center of mass is within the image bounds
        com = np.clip(com, [0, 0, 0], np.array(gradient_magnitude_map.shape) - 1)
        seeds_map[tuple(com)] = 1  # Assign a unique label to each seed

    return seeds_map




# TODO : Implement the watershed algorithm BIG TODO (for now it doesn't work)

def watershed_by_flooding(gradient_magnitude_map: np.ndarray,
                        seeds: np.ndarray,
                        mask: np.ndarray,
                        flooding_percent: float=85) -> np.ndarray:
    """
    Performs watershed segmentation by flooding, but restricts the flooding to the lowest 80% of gradient magnitude values.

    Args:
        gradient_magnitude_map (np.ndarray): The gradient magnitude map of the volume.
        seeds (np.ndarray): Seed points for the watershed algorithm.
        mask (np.ndarray): A mask to restrict the segmentation to a region of interest.
        flooding_percent (float): The percentage of lowest gradient magnitudes to flood.
    Returns:
        np.ndarray: The labeled volume after watershed segmentation.
    """
    gradient_magnitude_map = gradient_magnitude_map.copy()
    
    # Replace zero values with infinity to avoid issues
    gradient_magnitude_map[gradient_magnitude_map == 0] = np.inf

    # Compute the threshold corresponding to the 80th percentile of the lowest gradient magnitudes
    valid_values = gradient_magnitude_map[np.isfinite(gradient_magnitude_map)]
    threshold = np.percentile(valid_values, flooding_percent)

    # Create a mask for voxels to be processed (voxels with gradient magnitude <= threshold)
    process_mask = (gradient_magnitude_map <= threshold)

    # Initialize labels array
    labels, _ = ndimage.label(seeds > 0)
    
    # Initialize priority queue
    # Elements are tuples: (priority, x, y, z, label)
    heap = []
    
    # Create a status array to keep track of voxels
    # status == 0: unprocessed, status == 1: in queue, status == 2: processed
    status = np.zeros(labels.shape, dtype=np.uint8)
    status[labels > 0] = 2  # Mark seed points as processed

    # Get coordinates of all labeled seed points
    labeled_coords = np.argwhere(labels > 0)
    
    # For each labeled voxel, add its unprocessed neighbors to the queue
    for x, y, z in labeled_coords:
        current_label = labels[x, y, z]
        # Check 6-connected neighbors
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < labels.shape[0] and
                0 <= ny < labels.shape[1] and
                0 <= nz < labels.shape[2]):
                if process_mask[nx, ny, nz] and status[nx, ny, nz] == 0:
                    priority = gradient_magnitude_map[nx, ny, nz]
                    heapq.heappush(heap, (priority, nx, ny, nz, current_label))
                    status[nx, ny, nz] = 1  # Mark as in queue

    # Process the priority queue
    while heap:
        priority, x, y, z, current_label = heapq.heappop(heap)
        if status[x, y, z] == 2:
            continue  # Already processed
        if labels[x, y, z] == 0:
            labels[x, y, z] = current_label
        elif labels[x, y, z] != current_label:
            # Conflict detected, mark as boundary (-1)
            labels[x, y, z] = -1
        status[x, y, z] = 2  # Mark as processed
        
        # Add neighbors to queue
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < labels.shape[0] and
                0 <= ny < labels.shape[1] and
                0 <= nz < labels.shape[2]):
                if process_mask[nx, ny, nz] and status[nx, ny, nz] == 0:
                    neighbor_priority = gradient_magnitude_map[nx, ny, nz]
                    heapq.heappush(heap, (neighbor_priority, nx, ny, nz, labels[x, y, z]))
                    status[nx, ny, nz] = 1  # Mark as in queue
    
    # Apply the mask to remove labels outside the region of interest
    labels = labels * mask
    labels = labels.astype(np.int32) 
    return labels


if __name__ == "__main__":
    import os
    from datetime import datetime
    import nibabel as nib
    from toolbox_parcellation import extract_nii_files, expand_mask
    # Load basic data
    fmri_path = r'G:/DATA_min_preproc/dataset_study1/S02/wsraPPS-FACE_S02_005_Rest.nii' # fMRI data
    roi_mask_path = r'G:/MASK_standard/gm_postcentral_mask.nii' # Extraction of the grey matter in S1
    brain_mask_path = r'G:/MASK_standard/MNI152_T1_2mm_brain_mask.nii' # Whole brain mask MNI152 
    # Output file path
    output_dir = r'G:/DATA_min_preproc/dataset_study1/S02/'
    subject_id = r'S02'
    outdir_grad_map = output_dir + r'/outputs/edge_map'
    outdir_sim_mtrx = output_dir + r'/outputs/sim_mtrx'
    outdir_parcel =   output_dir + r'/outputs/parcels'
    fmri_data, roi_mask, brain_mask, original_affine = extract_nii_files(fmri_path, roi_mask_path, brain_mask_path, output_dir)
    extended_roi_mask = expand_mask(roi_mask, expansion_voxels=6)
    extended_roi_mask = brain_mask * extended_roi_mask # Ensure the mask is within the brain
    # Test the watershed by flooding algorithm from a gradient map
    outdir_grad_map = r'G:/DATA_min_preproc/dataset_study1/S02/outputs/edge_map'
    file_name = f"mean_edge_map_20241107_100132.nii"
    file_path = os.path.join(outdir_grad_map, file_name)
    nii_img = nib.load(file_path)
    mean_edge_map = nii_img.get_fdata()
    original_affine = nii_img.affine
    #%%
    seeds_map = find_seeds(mean_edge_map,
                           extended_roi_mask,
                           min_distance=2.5)
    labels_map = watershed_by_flooding(mean_edge_map, 
                                       seeds_map,
                                       extended_roi_mask,
                                       flooding_percent=100)
    print('number of seeds found : ', seeds_map.sum())  # Print the resulting labels
    print(f"Number of regions: {np.max(labels_map)}")  # Print the number of regions
    # %%
    # Save the parcellation map
    outdir_parcel = 'G:/DATA_min_preproc/dataset_study1/S02/outputs/parcels'
    # Save the parcellation map
    out_base_name = f'parcellation_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}'
    # Ensure the output directory exists
    os.makedirs(outdir_parcel, exist_ok=True)
    nii_img = nib.Nifti1Image(labels_map, affine=original_affine)
    nib.save(nii_img, os.path.join(outdir_parcel, out_base_name + '.nii'))

# import numpy as np
# from scipy.ndimage import minimum_filter, label, watershed_ift

# def find_local_minima(array: np.ndarray) -> np.ndarray:
#     local_minima = minimum_filter(array, size=3) == array
#     return local_minima

# def get_seeds(gradient_magnitude_map: np.ndarray) -> np.ndarray:
#     local_minima = find_local_minima(gradient_magnitude_map)
#     seeds = np.zeros_like(gradient_magnitude_map, dtype=np.uint32)
#     seeds[local_minima] = 1
#     return seeds

# def watershed(gradient_magnitude_map: np.ndarray) -> np.ndarray:
#     # Check if the array is of type uint32 and convert to uint16 for processing
#     if gradient_magnitude_map.dtype == np.uint32:
#         # Normalize the uint32 data to uint16 range (0-65535)
#         normalized_map = np.clip(gradient_magnitude_map, 0, 65535).astype(np.uint16)
#     else:
#         normalized_map = gradient_magnitude_map.astype(np.uint16)
    
#     # Generate seeds
#     seeds = get_seeds(normalized_map)
    
#     # Label the seed regions
#     markers, _ = label(seeds)
    
#     # Perform watershed using the labeled seeds and the normalized gradient map
#     watershed_map = watershed_ift(normalized_map, markers)
    
#     return watershed_map


# %%
