#%%
import os
from datetime import datetime
from typing import Literal
import numpy as np
import nibabel as nib
from sklearn.decomposition import PCA

from gradient import build_mesh_graph

def compute_similarity_matrix_pca(vol_fmri_n,
                                 surf_fmri_n,
                                 n_components=17):
    """Compute the similarity matrix with a PCA dim reduction
    Args:
        vol_fmri_n (_type_): _description_
        surf_fmri_n (_type_): _description_
        n_components (int, optional): _description_. Defaults to 17.
    """
    # 4. Run PCA on the time series data
    pca = PCA(n_components=n_components)
    temporal_modes = pca.fit_transform(vol_fmri_n.T).astype(np.float32) # shape = (n_timepoints, n_components)
    # print the percent of explained variacne 
    print('explained variance:', pca.explained_variance_ratio_)
    print('explained variance sum:', pca.explained_variance_ratio_.sum())
    
    # Correlation formula for normalized data
    corr_matrix = (surf_fmri_n @ temporal_modes)  / (surf_fmri_n.shape[1] - 1) # shape = (n_vertices, n_components)
    # Similarty matrix
    sim_matrix = np.corrcoef(corr_matrix, dtype=np.float32) # shape = (n_vertices, n_vertices)
    sim_matrix = np.nan_to_num(sim_matrix) # remove nans if any
    return sim_matrix


from sklearn.decomposition import FastICA
def compute_similarity_matrix_ica(surf_fmri_n,
                                 vol_fmri_n,
                                 mask_n,
                                 n_modes=7):
    X = vol_fmri_n[mask_n].T
    n_components = 7 # This is where you set the number of ICA sources
    ica = FastICA(n_components=n_components, random_state=0)
    temporal_modes = ica.fit_transform(X)   # S has shape (380, n_components)
    
    # compute the correlation matrix (for normalized data, the correlation matrix is the same as the covariance matrix)
    corr_matrix = (surf_fmri_n @ temporal_modes)  / (surf_fmri_n.shape[1] - 1)

    # make the similarty matrix as the correlation of the correlation matrix
    sim_matrix = np.corrcoef(corr_matrix)
    sim_matrix = np.nan_to_num(sim_matrix) # remove nans if any
    return sim_matrix



def save_similartiy_matrix(similarity_matrix,
                           output_dir,
                           hemisphere: Literal["lh", "rh"]
                           ) -> None:
    """Save the similarity matrix into a .npy file
    Args:
        gradient_map (2d np.array): the similartiy matrix (n_vertex, n_vertex) same order as coords
        output_dir (string): dir for sim map output
        hemisphere (strinf): the hemisphere of the surface data
    """
    time = datetime.now().strftime("%Y%m%d%H%M%S")
    path = output_dir + f"\{hemisphere}similarity_matrix_{time}.npy"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, similarity_matrix)

def load_similarity_matrix(path):
    """
    Load the similarity matrix
    """
    similarity_matrix = np.load(path)
    return similarity_matrix

#%% Run the code

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    import nibabel as nib 
    import networkx as nx
    from gradient import build_mesh_graph
    from preprocessing_surface import load_data_normalized
    
    config = {
    "fsavg6_dir": Path(r"D:\Data_Conn_Preproc\fsaverage6"),
    "subjects_dir": Path(r"D:\Data_Conn_Preproc\PPSFACE_N20")
    }
    run = 1
    subject = 1
    hemisphere ='lh'
    # Variables path
    subject = f"{subject:02d}"
    subj_dir = config["subjects_dir"] / f"sub-{subject}"
    
    surface_path = config["fsavg6_dir"] / "surf" / f"{hemisphere}.white"
    surface_inf_path = config["fsavg6_dir"] / "surf" / f"{hemisphere}.inflated"
    # Extract the Surface Mesh
    coords, faces = nib.freesurfer.read_geometry(str(surface_path))
    coords_, faces_ = nib.freesurfer.read_geometry(str(surface_inf_path))
    graph = build_mesh_graph(faces)
    
    # 1 : Compute the average similartiy matrix across each subjects
    sim_matrix_sum = np.zeros(nx.adjacency_matrix(graph).shape, dtype=np.float32)
    print('sim_matrix_sum shape', sim_matrix_sum.shape)

    # Define paths using pathlib
    surf_fmri_path = subj_dir / "func" / f"surf_conn_sub{subject}_run{run}_{hemisphere}.func.fsaverage6.mgh"
    vol_fmri_path = subj_dir / "func" / f"niftiDATA_Subject{subject}_Condition000_run{run}.nii.gz"
    brain_mask_path = subj_dir / f"sub{subject}_freesurfer" / "mri" / "brainmask.mgz"
    
    # Load the data
    surf_fmri_n, vol_fmri_n = load_data_normalized(surf_fmri_path,
                                                   vol_fmri_path, 
                                                   brain_mask_path)
    
    sim_matrix_sum = compute_similarity_matrix_pca(vol_fmri_n,
                                                   surf_fmri_n)
# %%
