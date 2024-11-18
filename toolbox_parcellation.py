#%%
import os
import sys
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt

def extract_4Ddata_from_nii(nii_file):
    """
    Extract data from nii file
    """
    try:
        print('Loading fmri from: ' + nii_file)
        Img = nib.load(nii_file)
        data = Img.get_fdata()
        original_affine = Img.affine
    except:
        sys.exit('Cannot open ' + nii_file | '\nExiting.')
    if len(data.shape) == 4:
        print(nii_file + ' is a 4D image\nExiting.')
    else:
        sys.exit('Data did not loaded successfully')
    return data, original_affine

def extract_3Ddata_from_nii(nii_file):
    """
    Extract data from nii file
    """
    try:
        print('Loading roi from: ' + nii_file)
        roiImg = nib.load(nii_file)
        original_affine = roiImg.affine
        data = roiImg.get_fdata()
    except:
        sys.exit('Cannot open ' + nii_file | '\nExiting.')
    if len(data.shape) == 3:
        print(nii_file + ' is a 3D image\nExiting.')
    else:
        sys.exit('Data did not loaded successfully')
    return data, original_affine


def extract_nii_files(fmri_path, roi_mask_path, brain_mask_path, output_dir):
    # Define file paths
    resampled_roi_mask_path = output_dir + 'resampled_roi_mask.nii'
    resampled_brain_mask_path = output_dir + 'resampled_brain_mask.nii'

    # Load images using nibabel
    fmri_img = nib.load(fmri_path)
    roi_mask_img = nib.load(roi_mask_path)
    brain_mask_img = nib.load(brain_mask_path)

    # Resample masks to fMRI space
    roi_mask_resampled = resample_to_img(
        source_img=roi_mask_img,
        target_img=fmri_img,
        interpolation='nearest'
    )

    brain_mask_resampled = resample_to_img(
        source_img=brain_mask_img,
        target_img=fmri_img,
        interpolation='nearest'
    )

    # Extract affine transformations
    fmri_affine = fmri_img.affine
    roi_affine = roi_mask_resampled.affine
    brain_affine = brain_mask_resampled.affine

    # Function to compare affines
    def affines_are_equal(affine1, affine2, tol=1e-5):
        return np.allclose(affine1, affine2, atol=tol)

    # Verify that all affines match the fMRI affine
    roi_affine_matches = affines_are_equal(fmri_affine, roi_affine)
    brain_affine_matches = affines_are_equal(fmri_affine, brain_affine)

    print(f"ROI Mask affine matches fMRI affine: {roi_affine_matches}")
    print(f"Brain Mask affine matches fMRI affine: {brain_affine_matches}")

    # Assert if affines do not match
    assert roi_affine_matches, "ROI mask affine does not match fMRI affine."
    assert brain_affine_matches, "Brain mask affine does not match fMRI affine."
    # Optional: Save resampled masks
    os.makedirs(output_dir, exist_ok=True)
    roi_mask_resampled.to_filename(resampled_roi_mask_path)
    brain_mask_resampled.to_filename(resampled_brain_mask_path)

    # Convert to np array
    fmri_array = fmri_img.get_fdata()
    roi_mask_array = roi_mask_resampled.get_fdata()
    brain_mask_array = brain_mask_resampled.get_fdata()

    return fmri_array, roi_mask_array, brain_mask_array, fmri_affine



def pca(X):
    from scipy.linalg import svd
    # Center X by subtracting off column means
    X -= np.mean(X,0)
    # The principal components are the eigenvectors of S = X'*X./(n-1), but computed using SVD
    [U,sigma,V] = svd(X,full_matrices=False)
    # Project X onto the principal component axes
    Y = U*sigma
    # Convert the singular values to eigenvalues 
    sigma /= np.sqrt(X.shape[0]-1)
    evals = np.square(sigma)
    
    return V, Y, evals

def corr(X,Y):
    Y = Y.T
    X = X.T
    R = np.zeros((X.shape[0],Y.shape[0]))
    for i in range(0,R.shape[1]):
        y = Y[i,:]
        Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
        ym = np.mean(y)
        r_num = np.sum((X-Xm)*(y-ym),axis=1)
        r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
        R[:,i] = r_num / r_den
    return R

def eta2(X):
    
    S = np.zeros((X.shape[0],X.shape[0]))
    for i in range(0,X.shape[0]):
        for j in range(i,X.shape[0]):
            mi = np.mean([X[i,:],X[j,:]],0) 
            mm = np.mean(mi)
            ssw = np.sum(np.square(X[i,:]-mi) + np.square(X[j,:]-mi))
            sst = np.sum(np.square(X[i,:]-mm) + np.square(X[j,:]-mm))
            S[i,j] = 1-ssw/sst
    
    S += S.T 
    S -= np.eye(S.shape[0])
    
    return S



#%%

def create_boundary_maps(parcellation):
    """This function creates the boundary maps from a parcellation data.
    Args:
        parcellation (3D np.array): 3D numpy array representing the parcellation data
        The values are integers representing the different parcels.
    Returns:
        3D np.array: 3D numpy array representing the boundary map.
    """ 
    #Initialize the boundary map with zeros
    boundary_map = np.zeros_like(parcellation, dtype=int)
    
    # Get the shape of the parcellation
    x_dim, y_dim, z_dim = parcellation.shape
    
    # Iterate through each voxel in the 3D array (excluding the edges to avoid out-of-bounds access)
    for x in range(1, x_dim - 1):
        for y in range(1, y_dim - 1):
            for z in range(1, z_dim - 1):
                # Current voxel value
                current_value = parcellation[x, y, z]
                
                # Check the six neighbors (left, right, front, back, above, below)
                neighbors = [
                    parcellation[x - 1, y, z],  # left
                    parcellation[x + 1, y, z],  # right
                    parcellation[x, y - 1, z],  # front
                    parcellation[x, y + 1, z],  # back
                    parcellation[x, y, z - 1],  # below
                    parcellation[x, y, z + 1]   # above
                ]
                
                # If the current voxel value differs from any of its neighbors, it's a boundary
                if any(current_value != neighbor for neighbor in neighbors):
                    boundary_map[x, y, z] = 1
    return boundary_map


# This part is used to download data from the HCP
def expand_mask(mask, expansion_voxels=2):
    """
    Expands a 3D binary mask by a specified number of voxels in all directions.
    Parameters:
    - mask (np.ndarray): A 3D NumPy array with binary values (0 and 1).
    - expansion_voxels (int): Number of voxels to expand the mask in each direction.

    Returns:
    - expanded_mask (np.ndarray): The expanded 3D binary mask.
    """
    from scipy.ndimage import binary_dilation
    # Validate input
    if not isinstance(mask, np.ndarray):
        raise TypeError("Input mask must be a NumPy array.")
    if mask.ndim != 3:
        raise ValueError("Input mask must be a 3D array.")
    if not isinstance(expansion_voxels, int) or expansion_voxels < 0:
        raise ValueError("expansion_voxels must be a non-negative integer.")
    # Perform binary dilation
    expanded_mask = binary_dilation(mask, iterations=expansion_voxels).astype(mask.dtype)
    return expanded_mask


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
    

#%% BULLSHIT CODE IN ORDER FOR ME TO UNDERSTAAND THE THEORY
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from scipy.sparse import csgraph
# from sklearn.datasets import make_moons

# from sklearn.metrics import pairwise_distances
# # Step 1: Generate a synthetic graph with a nontrivial structure for clustering
# # We'll use "make_moons" to create a dataset that has two interleaved clusters
# X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)


# # Step 1: Compute the Euclidean distance matrix
# distance_matrix = pairwise_distances(X, metric='euclidean')

# # Step 2: Define the adjacency matrix based on a distance threshold
# threshold = 0.3 # Define a threshold (tune this based on data spread)
# adjacency_matrix = np.divide(1, distance_matrix, where=distance_matrix != 0)*(distance_matrix < threshold).astype(float)  # Binary adjacency matrix
# # Step 3: Construct the degree matrix and Laplacian
# degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
# laplacian = degree_matrix - adjacency_matrix

# # Step 4: Compute the smallest 2 eigenvalues and eigenvectors of the Laplacian for 2D embedding
# eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
# embedding = eigenvectors[:, 1:3]  # Skip the first (smallest) eigenvector

# # Step 5: Apply k-means clustering to the 2D embedding
# kmeans = KMeans(n_clusters=2, random_state=42)
# clusters = kmeans.fit_predict(embedding)

# # Visualization
# fig, ax = plt.subplots(1, 3, figsize=(12, 5))

# # Original data visualization
# ax[0].scatter(X[:, 0], X[:, 1], c='gray', edgecolor='k', s=50)
# ax[0].set_title("Original Data")
# ax[0].set_xlabel("X-axis")
# ax[0].set_ylabel("Y-axis")

# # Clustering result in the embedded space (2D Laplacian Eigenmap)
# ax[1].scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', s=50, edgecolor='k')
# ax[1].set_title("Clustering in 2D Laplacian Eigenmap Space")
# ax[1].set_xlabel("Eigenvector 1")
# ax[1].set_ylabel("Eigenvector 2")

# # Original data visualization with clusters
# ax[2].scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
# ax[2].set_title("Original Data with Clusters")
# ax[2].set_xlabel("X-axis")
# ax[2].set_ylabel("Y-axis")

# plt.tight_layout()
# plt.show()

# %%
