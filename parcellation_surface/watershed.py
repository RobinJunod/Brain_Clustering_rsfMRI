"""
Watershed by flooding algorithm for surface mesh segmentation.
"""
import os
from typing import Literal
from datetime import datetime
import heapq
import numpy as np
import nibabel as nib


def find_local_minima(values, graph):
    """
    Args:
        values: (N,) array-like of scalar values (gradient magnitude at each vertex).
        graph: networkx.Graph where each node corresponds to an index in [0..N-1].
    
    Returns:
        minima: list of vertex indices that are local minima
    """
    minima = []
    for node in graph.nodes():
        val = values[node]
        # Check all neighbors
        is_min = True
        for nbr in graph.neighbors(node):
            if values[nbr] <= val:  
                # <= means we require strictly smaller for "is_min" to remain True
                # (use < if you want to allow equals)
                is_min = False
                break
        if is_min:
            minima.append(node)
    return minima

def watershed_by_flooding(graph, values):
    """
    Perform a watershed segmentation on 'values' defined on a mesh graph.
    
    Args:
        graph: networkx.Graph where each node is an int in [0..N-1].
        values: (N,) array of floats (e.g., gradient magnitude at each vertex).
        
    Returns:
        labels : (N,) array of integers
            -2 = boundary vertex
            -1 = unassigned vertex
            >= 0 = index of the "basin" region
    """
    n_vertices = graph.number_of_nodes()
    # We'll assume nodes go from 0..(n_vertices-1)
    
    # 1) Find local minima
    minima_indices = find_local_minima(values, graph)
    
    # labels[i] indicates:
    #   -2 => boundary
    #   -1 => unassigned
    #   >=0 => basin ID
    labels = np.full(n_vertices, -1, dtype=int)
    
    # 2) Assign each local minimum a unique label
    for basin_id, idx in enumerate(minima_indices):
        labels[idx] = basin_id

    # 3) Initialize a priority queue (value, vertex_index)
    pq = []
    for i in minima_indices:
        heapq.heappush(pq, (values[i], i))

    # 4) Flooding
    while pq:
        current_value, current_vertex = heapq.heappop(pq)
        
        current_label = labels[current_vertex]
        if current_label < 0:
            # If it's unassigned or changed, skip
            continue
        if current_label == -2:
            # Already boundary => do not flood from here
            continue
        
        # Check neighbors
        for nbr in graph.neighbors(current_vertex):
            nbr_label = labels[nbr]
            
            if nbr_label == -1:
                # unassigned => adopt current_vertex's label
                labels[nbr] = current_label
                # push neighbor into the queue
                heapq.heappush(pq, (values[nbr], nbr))
            
            elif nbr_label >= 0 and nbr_label != current_label:
                # Conflict => different basin => mark boundary
                labels[current_vertex] = -2
                labels[nbr] = -2
                # Once marked boundary, we stop flooding from current_vertex
                # You can break or continue depending on how you want to handle
                break
    
    return labels



##################### Alternative method for edges detection #####################
# Non-maxima suppression
def non_maxima_suppression(graph,
                           gradient_map,
                           min_neighbors=4,
                           threshold=0.7):
    """
    Apply non-maxima suppression to identify edge vertices.
    
    Args:
        graph: networkx.Graph object
        gradient_map: ndarray of shape (n_vertices,)
        min_neighbors: int, number of non-adjacent maxima required
    
    Returns:
        edge_map: ndarray of shape (n_vertices,), boolean
    """
    # The minimum gradient value for a local maximum
    min_grad_value = np.percentile(gradient_map, 100*threshold)
    # Initialize the edge map
    edge_map = np.zeros_like(gradient_map, dtype=bool)
    for vertex in graph.nodes:
        neighbors = list(graph.neighbors(vertex))
        if len(neighbors) < min_neighbors:
            continue
        # Check if current vertex is a local maximum
        is_max = True
        count = 0
        for neighbor in neighbors:
            if gradient_map[vertex] <= gradient_map[neighbor]:
                is_max = False
                break
            count += 1
            if count >= min_neighbors:
                break
        if is_max and count >= min_neighbors and gradient_map[vertex] > min_grad_value:
            edge_map[vertex] = True
    return edge_map




def save_labels_mgh(labels,
                    output_dir,
                    hemisphere: Literal["lh", "rh"],
                    name = "labels"):
    """Save the gradient map into a .mgh file
    Args:
        labels (np.array): the labels in order to the coords from the triangles surface
        output_dir (string): dir for grad output
        hemisphere (strinf): the hemisphere of the surface data
    """
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    # Reshape the data to match the FreeSurfer .mgh format expectations
    labels_reshaped = labels.reshape((len(labels), 1, 1)).astype(np.float32)
    # 3. Create an identity affine (often used for surface data).
    affine = np.eye(4)
    # 4. Construct the MGH image.
    mgh_img = nib.freesurfer.mghformat.MGHImage(labels_reshaped, affine)
    # 5. Save the MGH file to disk.
    path = output_dir + f"\{name}_{hemisphere}_{time}.mgh"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(mgh_img, path)

def load_labels_mgh(mgh_file_path):
    # Load the MGH image
    mgh_image = nib.load(mgh_file_path)
    # Extract data as float32 (optional) and squeeze to remove single-dimensional axes
    data = mgh_image.get_fdata().squeeze()
    return data



