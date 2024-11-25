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
# %%
