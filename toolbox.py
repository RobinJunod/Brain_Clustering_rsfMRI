#%% BULLSHIT CODE IN ORDER FOR ME TO UNDERSTAAND THE THEORY
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
from sklearn.datasets import make_moons, make_circles, make_blobs

from sklearn.metrics import pairwise_distances
# Step 1: Generate a synthetic graph with a nontrivial structure for clustering
# We'll use "make_moons" to create a dataset that has two interleaved clusters
# X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)


# Step 1: Compute the Euclidean distance matrix
distance_matrix = pairwise_distances(X, metric='euclidean')

# Step 2: Define the adjacency matrix based on a distance threshold
threshold = 0.3 # Define a threshold (tune this based on data spread)
adjacency_matrix = np.divide(1, distance_matrix, where=distance_matrix != 0)*(distance_matrix < threshold).astype(float)  # Binary adjacency matrix
# Step 3: Construct the degree matrix and Laplacian
degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
laplacian = degree_matrix - adjacency_matrix

# Step 4: Compute the smallest 2 eigenvalues and eigenvectors of the Laplacian for 2D embedding
eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
embedding = eigenvectors[:, 1:3]  # Skip the first (smallest) eigenvector

# Step 5: Apply k-means clustering to the 2D embedding
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embedding)
# Step 5.0: Apply clustering unsupervised learning algorithm with spectral clustering
fielder_vector = eigenvectors[:, 1]
clusters = fielder_vector > 0  # Cluster based on the sign of the second eigenvector


# Visualization
fig, ax = plt.subplots(2, 2, figsize=(12, 5))

# Original data visualization
ax[0,0].scatter(X[:, 0], X[:, 1], c='gray', edgecolor='k', s=50)
ax[0,0].set_title("Original Data")
ax[0,0].set_xlabel("X-axis")
ax[0,0].set_ylabel("Y-axis")

# Clustering result in the embedded space (2D Laplacian Eigenmap)
ax[1,0].plot(np.sort(fielder_vector))
ax[1,0].set_title("Fielder vector")
ax[1,0].set_xlabel("Indices")
ax[1,0].set_ylabel("Sorted Eigenvector 1")

# Clustering result in the embedded space (2D Laplacian Eigenmap)
ax[0,1].scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', s=50, edgecolor='k')
ax[0,1].set_title("Clustering in 2D Laplacian Eigenmap Space")
ax[0,1].set_xlabel("Eigenvector 1")
ax[0,1].set_ylabel("Eigenvector 2")
# Original data visualization with clusters
ax[1,1].scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
ax[1,1].set_title("Original Data with Clusters")
ax[1,1].set_xlabel("X-axis")
ax[1,1].set_ylabel("Y-axis")

plt.tight_layout()
plt.show()

# %%

fig, ax = plt.subplots(figsize=(12, 5))
scatter = ax.scatter(X[:, 0], X[:, 1], c=fielder_vector, cmap='viridis', s=50, edgecolor='k')
ax.set_title("Original Data Colored by Fiedler Vector")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
fig.colorbar(scatter, ax=ax, label="Fiedler Vector Value")

plt.tight_layout()
plt.show()




# %% SPLIT A NII FILE INTO TWO PARTS
import nibabel as nib
import numpy as np

def split_nii_file(nii_path,
                   sub,
                   split_index=380,
                   output_prefix="conn_wsraPPSFACE_sub-",
                   ):
    """
    Splits a 4D NIfTI file into two separate NIfTI files at the given timestep.

    Args:
        nii_path (str): Path to the input .nii or .nii.gz file.
        split_index (int): The timestep at which to split the data.
        output_prefix (str): Prefix for output filenames.
    """
    # Load NIfTI file
    nii = nib.load(nii_path)
    data = nii.get_fdata()  # Convert to NumPy array
    affine = nii.affine
    header = nii.header
    output_prefix = output_prefix + sub
    # Ensure it's a 4D NIfTI file (X, Y, Z, T)
    if len(data.shape) != 4:
        raise ValueError("The NIfTI file must be 4D (X, Y, Z, T).")

    # Split the data along the time axis (last axis)
    data_part1 = data[..., :split_index]  # First half (up to timestep 380)
    data_part2 = data[..., split_index:]  # Second half (from timestep 380)

    # Save the first part
    nii_part1 = nib.Nifti1Image(data_part1, affine, header)
    nib.save(nii_part1, f"{output_prefix}_run1.nii.gz")

    # Save the second part
    nii_part2 = nib.Nifti1Image(data_part2, affine, header)
    nib.save(nii_part2, f"{output_prefix}_run2.nii.gz")

    print(f"Files saved as {output_prefix}_run1.nii.gz and {output_prefix}_run2.nii.gz")
#%%
for i in range(2,11):
    sub = "%02d" % i
    # Example usage
    split_nii_file(r"D:\Data_Conn_Preproc\PPSFACE_N18\niftiDATA_Subject0"+"01"+r"_Condition000.nii",
                   '01',
                   split_index=380)


# %%
