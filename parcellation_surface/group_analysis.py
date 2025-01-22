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
from smoothing import smooth_surface
from watershed import watershed_by_flooding
from gradient import build_mesh_graph
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
    for s in range(1,19): # TODO : customize the range
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
        parcel_data[i] = np.mean(surf_fmri[np.where(group_parcel==i)], axis=0)
        print(np.isnan(parcel_data[i]).any())
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
    # surface_path = r"D:\DATA_min_preproc\dataset_study1\fsaverage6\surf\lh.white"
    surface_path = r"D:\DATA_min_preproc\dataset_study1\fsaverage6\surf\lh.inflated"
    coords, faces = nib.freesurfer.read_geometry(surface_path)
    lh_fmri_path = r"D:\DATA_min_preproc\dataset_study1\sub-01\func\sub01_lh.func.fsaverage6.mgh"
    # rh_fmri_path = r"D:\DATA_min_preproc\dataset_study1\sub-01\func\sub01_rh.func.fsaverage6.mgh"
    surf_fmri_img = nib.load(lh_fmri_path)
    surf_fmri = surf_fmri_img.get_fdata()
    
    gradient_list, parcel_list = extracrt_gradparc_list(hemisphere='lh')
    # print(group_gradient.shape)
    # print(group_parc_bound.shape)
    visualize_brain_surface(coords, faces, gradient_list[0], title="Group Gradient")
    
    # Compute group parcellation
    group_gradient = gradient_list.mean(axis=0)
    group_gradient_smoothed = smooth_surface(faces, group_gradient,  iterations=10)
    visualize_brain_surface(coords, faces, group_gradient_smoothed, title="Group Gradient")
    # Labels the group gradient map
    graph = build_mesh_graph(faces)
    group_parc = watershed_by_flooding(graph, group_gradient_smoothed)
    group_boundary = (group_parc<0)*1
    visualize_brain_surface(coords, faces, group_boundary, title="Group Bounardy")
    
    # Compute dice coefficient
    dice_coef_list = []
    for i in range(18):
        dice = dice_coefficient(group_parc, parcel_list[i])
        dice_coef_list.append(dice)
        print(f"Subject {i+1} Dice coefficient: {dice}")
        
    # Compare group dice coefficient
    groupA_mean = gradient_list[:8,:].mean(axis=0)
    groupA_mean = smooth_surface(faces, groupA_mean,  iterations=10)
    groupA_parc = watershed_by_flooding(graph, groupA_mean)
    groupA_boundary = (groupA_parc<0)*1
    
    groupB_mean = gradient_list[8:,:].mean(axis=0)
    groupB_mean = smooth_surface(faces, groupB_mean,  iterations=10)
    groupB_parc = watershed_by_flooding(graph, groupB_mean)
    groupB_boundary = (groupB_parc<0)*1
    
    #%% Load all of the surf fmri data
    surf_fmri_list = []
    dataset_dir = r"D:\DATA_min_preproc\dataset_study1"
    for i in range(1,19):
        subject = f"{i:02d}"
        subj_dir = dataset_dir + r"\sub-" + subject
        lh_fmri_path = subj_dir + r"\func\sub" + subject + "_lh.func.fsaverage6.mgh"
        surf_fmri_img = nib.load(lh_fmri_path)
        surf_fmri = surf_fmri_img.get_fdata()
        surf_fmri = np.squeeze(surf_fmri)
        surf_fmri_list.append(surf_fmri)
    
    sub1_parccorr = parcel_correlation(group_parc, surf_fmri_list[0])
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))  # Adjust figure size for better visibility
    sns.heatmap(
        sub1_parccorr, 
        cmap="coolwarm",  # Colormap to highlight intensity
        annot=False,  # Set to True if you want values displayed in the cells
        fmt=".2f",  # Format for annotations (if enabled)
        cbar=True  # Show color bar
    )
    plt.title("Connectivity of Subj1 with parcels", fontsize=16)
    plt.xlabel("Parcel idx")
    plt.ylabel("Parcel idx")
    plt.show()
# %%

# # A boxplot fucntion with individual data points
# import seaborn as sns
# import matplotlib.pyplot as plt
# # Set the style for a cleaner look
# sns.set(style="whitegrid", palette="muted")
# # Create the plot
# values = dice_coef_list
# plt.figure(figsize=(3, 5))
# # Create a boxplot with more elegant styling
# sns.boxplot(data=values, color='skyblue', width=0.5, fliersize=8, linewidth=2, 
#             boxprops=dict(facecolor='skyblue', edgecolor='black'),
#             whiskerprops=dict(color='black', linewidth=1.5),
#             capprops=dict(color='black', linewidth=1.5),
#             medianprops=dict(color='darkred', linewidth=2))

# # Overlay individual data points with customized aesthetics
# sns.stripplot(data=values, color='darkblue', jitter=True, size=8, edgecolor='white', linewidth=1)
# # Add title and labels
# plt.title('Single subject parcellation Vs Group average', fontsize=16, fontweight='bold')
# plt.ylabel('Dice coefficient', fontsize=14)
# plt.xlabel('', fontsize=12)  # Remove x-axis label
# # Remove the x-axis ticks as there is only one group
# plt.xticks([])
# # Customize grid lines for a cleaner look
# plt.grid(True, axis='y', linestyle='--', alpha=0.7)
# # Show the plot
# plt.tight_layout()
# plt.show()



# IDEA TO IMPLEMENT IN ORDER TO IMPROVE THE PARCELLATION NAD RESULTS FOR THE GRADIETN MAP
# FILTER THE SIMILARITY MAP TO HIGHLIGHT HIGH SIMILARTIY AREA AND EXTRACT THIS BOUNDARY

# %%
