"""
Module Name: group_analysis.py
Description:
    This script demonstrates how to load and process surface data from Freesurfer outputs.
    It includes utilities for:
      - Extracting timestamps from file names
      - Extracting gradient and parcellation maps
      - Computing Dice coefficients
      - Performing correlations between parcel data

Author: Robin Junod, robin.junod@epfl.ch
Created: 2025-01-16

Usage:
    python group_analysis.py
"""

#%%
import os
import glob
import numpy as np
from typing import Literal # Requires Python 3.8+
from visualization import visualize_brain_surface
from gradient import load_gradient_mgh
from watershed import load_labels_mgh



def extract_timestamp(fpath):
    fname = os.path.basename(fpath)  # e.g. "left_labels_20240101123045.npy"
    # Split by "_" -> ["left", "labels", "20240101123045.npy"]
    # The last part has "20240101123045.npy"
    time_str = fname.split('_')[-1].replace(".npy", "")
    return time_str

def extracrt_gradparc_list(hemisphere: Literal["lh", "rh"],
                           dataset_dir = r"D:\DATA_min_preproc\dataset_study1"
                           ):
    list_parc = []
    list_grad = []
    for s in range(1,4): # TODO : customize the range
        # print('grad map and parcellation map of subject : ', s)
        subject = f"{s:02d}"
        # Path to the subject directory
        subj_dir = dataset_dir + r"\sub-" + subject
        grad_dir = subj_dir + r"\outputs_surface\gradient_map_fsavg6_highsmooth"
        parcel_dir = subj_dir + r"\outputs_surface\labels_fsavg6_highsmooth"
        pattern_grad = os.path.join(grad_dir, f"{hemisphere}_gradient_map_*.mgh")
        pattern_parc = os.path.join(parcel_dir, f"{hemisphere}_labels_*.mgh")

        files_grad = glob.glob(pattern_grad)
        files_parc = glob.glob(pattern_parc)

        # Select latest files 
        files_grad_sorted = sorted(files_grad, key=lambda x: extract_timestamp(x))
        files_parc_sorted = sorted(files_parc, key=lambda x: extract_timestamp(x))

        latest_grad_file = files_grad_sorted[-1]
        latest_parc_file = files_parc_sorted[-1]
        
        # Add the data to the list 
        list_grad.append(load_gradient_mgh(latest_grad_file))
        
        # extract the boundary of the parcellation map
        parc_boundary = load_labels_mgh(latest_parc_file)
        parc_boundary = 1*(parc_boundary<0)
        list_parc.append(parc_boundary)
    
    gradient_list = np.stack(list_grad, axis=0)
    # group_gradient = np.mean(np.stack(list_grad, axis=0), axis=0)
    parcel_list = np.stack(list_parc, axis=0)
    # parcels_boundaries = (group_parcel_list < 0)*1
    # group_parc_bound = np.mean(parcels_boundaries, axis=0)
    # extract the edges of the group parcellation
    return gradient_list, parcel_list

def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice coefficient between two binary masks.
    
    Args:
        y_true : (N,) ndarray
            Ground-truth binary mask.
        y_pred : (N,) ndarray
            Predicted binary mask.
    
    Returns:
        float : Dice coefficient.
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    if union == 0:
        return 1.0
    return 2 * intersection / union

def parcel_correlation(group_parcel, surf_fmri):
    """
    Group the fMRI data into parcels and compute the correlation between parcels.
        group_parcel (np.ndarray): An array where each element indicates the parcel assignment 
                                   of the corresponding vertex in the surface fMRI data.
        surf_fmri (np.ndarray): A 2D array where each row represents a vertex and each column 
                                represents a time point of the fMRI data.
    Returns:
        np.ndarray: A 2D array representing the correlation matrix between the parcels.
    """
    # Group the fmri data into parcels
    n_parcels = np.max(group_parcel) + 1
    parcel_data = np.zeros((n_parcels, surf_fmri.shape[1]))
    for i in range(n_parcels):
        parcel_data[i] = np.mean(surf_fmri[group_parcel == i], axis=0)
    # Compute the correlation between parcels
    parcel_corr = np.corrcoef(parcel_data)
    return parcel_corr

def corr2graph(correlation_matrix):
    """Create a nx.Graph object from a correlation matrix.
    Each edge weight is the correlation between the two vertices.
    Args:
        correlation_matrix (np.ndarray): A square matrix of correlations.
    """
    import networkx as nx
    # The 2D NumPy array is interpreted as an adjacency matrix for the graph.
    G = nx.from_numpy_array(correlation_matrix)
    return G


    
    

# TODO : try to visualize the group gradient and the group parcellation
#%% TODO : try to perform watershed on gradient and on parcels
if __name__ == "__main__":
    import nibabel as nib
    surface_path = r"D:\DATA_min_preproc\dataset_study1\fsaverage6\surf\lh.white"
    coords, faces = nib.freesurfer.read_geometry(surface_path)
    gradient_list, parcel_list = extracrt_gradparc_list(hemisphere='lh')
    # print(group_gradient.shape)
    # print(group_parc_bound.shape)
    # visualize_brain_surface(coords, faces, group_gradient, title="Group Gradient")
# %%

