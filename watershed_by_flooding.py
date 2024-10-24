"""This file has an implementation of the watershed by flooding algorithm for 3D dataset. 
The watershed algorithm is a classical algorithm used for image segmentation.

This algorithm needs a gradient magnitude map.

@myspace 2024-2025 EPFL
"""
#%%
import numpy as np
import heapq
from scipy import ndimage


def find_seeds(gradient_magnitude_map: np.ndarray) -> np.ndarray:
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


# TODO : Implement the watershed algorithm BIG TODO (for now it doesn't work)

def watershed_by_flooding(gradient_magnitude_map: np.ndarray) -> np.ndarray:

    print('Search seeds for the watershed algorithm...')
    seeds_maps = find_seeds(gradient_magnitude_map)
    
    print('Performing watershed by flooding...')
    # gradient map preprocessing
    indices = np.where(gradient_magnitude_map==0)
    gradient_magnitude_map[indices] = np.inf
    # create mask
    mask = np.ones(gradient_magnitude_map.shape)
    mask[indices] = 0
    
    # Initialize labels array
    labels, num_labels = ndimage.label(seeds_maps > 0)
    
    # Initialize priority queue
    # Elements are tuples: (priority, x, y, z, label)
    heap = []
    
    # Create a status array to keep track of pixels
    # status == 0: unprocessed, status == 1: in queue, status == 2: processed
    status = np.zeros(labels.shape, dtype=np.uint8)
    status[labels > 0] = 2  # Mark seed points as processed

    # Get coordinates of all labeled seed points
    labeled_coords = np.argwhere(labels > 0)
    
    # For each labeled pixel, add its unlabeled neighbors to the queue
    for x, y, z in labeled_coords:
        current_label = labels[x, y, z]
        # Check 6-connected neighbors
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < labels.shape[0] and
                0 <= ny < labels.shape[1] and
                0 <= nz < labels.shape[2]):
                if status[nx, ny, nz] == 0:
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
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < labels.shape[0] and
                0 <= ny < labels.shape[1] and
                0 <= nz < labels.shape[2]):
                if status[nx, ny, nz] == 0:
                    heapq.heappush(heap, (gradient_magnitude_map[nx, ny, nz], nx, ny, nz, labels[x, y, z]))
                    status[nx, ny, nz] = 1  # Mark as in queue
    
    # Remove wrong labels
    labels = labels * mask
    return labels


if __name__ == "__main__":
    import os
    import nibabel as nib
    # Test the watershed by flooding algorithm from a gradient map
    outdir_grad_map ='G:/HCP/outputs/grad_maps/'
    file_name = "gradient_map_S01_20241021_163908.nii"
    file_path = os.path.join(outdir_grad_map, file_name)
    nii_img = nib.load(file_path)
    gradient_magnitude_map = nii_img.get_fdata()
    original_affine = nii_img.affine
    #%%
    labels_map = watershed_by_flooding(gradient_magnitude_map)
    print(labels_map)  # Print the resulting labels
    print(f"Number of regions: {np.max(labels_map)}")  # Print the number of regions