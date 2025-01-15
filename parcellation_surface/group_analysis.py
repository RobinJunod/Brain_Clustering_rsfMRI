#%%
import os
import glob
import numpy as np
from typing import Literal # Requires Python 3.8+

from gradient import build_mesh_graph, compute_gradients
from watershed import watershed_by_flooding
from visualization import visualize_brain_surface




def extract_timestamp(fpath):
    fname = os.path.basename(fpath)  # e.g. "left_labels_20240101123045.npy"
    # Split by "_" -> ["left", "labels", "20240101123045.npy"]
    # The last part has "20240101123045.npy"
    time_str = fname.split('_')[-1].replace(".npy", "")
    return time_str

def extracrt_gradparc_list(hemisphere: Literal["lh", "rh"]):
    list_parc = []
    list_grad = []
    dataset_dir = r"D:\DATA_min_preproc\dataset_study1"
    for s in range(1,19): # TODO : customize the range
        # print('grad map and parcellation map of subject : ', s)
        subject = f"{s:02d}"
        # Path to the subject directory
        subj_dir = dataset_dir + r"\sub-" + subject
        grad_dir = subj_dir + r"\outputs_surface\gradient_map"
        parcel_dir = subj_dir + r"\outputs_surface\labels"
        pattern_grad = os.path.join(grad_dir, f"{hemisphere}_gradient_map_*.npy")
        pattern_parc = os.path.join(parcel_dir, f"{hemisphere}_labels_*.npy")

        files_grad = glob.glob(pattern_grad)
        files_parc = glob.glob(pattern_parc)

        # Select latest files 
        files_grad_sorted = sorted(files_grad, key=lambda x: extract_timestamp(x))
        files_parc_sorted = sorted(files_parc, key=lambda x: extract_timestamp(x))

        latest_grad_file = files_grad_sorted[-1]
        latest_parc_file = files_parc_sorted[-1]
        
        # Add the data to the list 
        list_grad.append(np.load(latest_grad_file))
        
        # extract the boundary of the parcellation map
        parc_boundary = np.load(latest_parc_file)
        parc_boundary = 1*(parc_boundary<0)
        list_parc.append(parc_boundary)
    
    group_gradient = np.mean(np.stack(list_grad, axis=0), axis=0)
    group_parcel = np.sum(np.stack(list_parc, axis=0), axis=0)
    return group_gradient, group_parcel


def compute_average_midthickness(surf_paths, out_path):
    """
    Given a list of GIFTI surface paths (midthickness from multiple subjects),
    compute the average coordinates and save a new GIFTI file.
    """
    all_coords = []
    common_faces = None

    for i, path in enumerate(surf_paths):
        gii = nib.load(path)
        coords = gii.darrays[0].data  # shape: (N_vertices, 3)
        faces = gii.darrays[1].data   # shape: (N_faces, 3)
        
        # Save faces from the first file (assuming all are the same).
        if i == 0:
            common_faces = faces

        # We assume faces are the same for all subjects—this must be verified.
        # If not the same, we need a different approach (surface registration).
        all_coords.append(coords)

    # Stack into (N_subjects, N_vertices, 3) and average across subjects
    all_coords = np.stack(all_coords, axis=0)
    avg_coords = np.mean(all_coords, axis=0)  # (N_vertices, 3)

    # Build a new GIFTI image
    # Each darray must be a GiftiDataArray; first is coords, second is faces.
    coord_gda = nib.gifti.GiftiDataArray(data=avg_coords.astype(np.float32),
                                         intent='NIFTI_INTENT_POINTSET')
    face_gda = nib.gifti.GiftiDataArray(data=common_faces.astype(np.int32),
                                        intent='NIFTI_INTENT_TRIANGLE')

    gii_out = nib.gifti.GiftiImage(darrays=[coord_gda, face_gda])
    nib.save(gii_out, out_path)

    print(f"Saved group-average midthickness to: {out_path}")
    
    
    

# TODO : try to visualize the group gradient and the group parcellation
# TODO : try to perform watershed on gradient and on parcels
#%%
if __name__ == "__main__":
    import nibabel as nib
    dataset_dir = r"D:\DATA_min_preproc\dataset_study1"
    
    # Compute the group average midthickness
    surf_paths = [f"{dataset_dir}/sub-{i:02d}/sub{i:02d}_freesurfer/surf/lh.midthickness.32k.surf.gii" for i in range(1, 19)]
    out_path = f"{dataset_dir}/lh.group_average_midthickness.gii"
    compute_average_midthickness(surf_paths, out_path)
    
    # Compute the group average midthickness right hemisphere
    surf_paths = [f"{dataset_dir}/sub-{i:02d}/sub{i:02d}_freesurfer/surf/rh.midthickness.32k.surf.gii" for i in range(1, 19)]
    out_path = f"{dataset_dir}/rh.group_average_midthickness.gii"
    compute_average_midthickness(surf_paths, out_path)
    
    
    # Compute the group average gradient and parcellation
    for hemisphere in ["lh", "rh"]:
        group_gradient, group_parcel = extracrt_gradparc_list(hemisphere=hemisphere)
        # Save the group gradient and parcellation
        np.save(dataset_dir + "\\" + hemisphere + ".group_gradient.npy", group_gradient)
        np.save(dataset_dir + "\\" + hemisphere + ".group_parcel.npy", group_parcel)

    # Build the mesh graph
    path_lh_midthickness_mean = dataset_dir + r"\lh.group_average_midthickness.gii"
    gii = nib.load(path_lh_midthickness_mean)
    coords = gii.darrays[0].data  # shape: (N_vertices, 3)
    faces = gii.darrays[1].data   # shape: (N_faces, 3)
    values = np.load(dataset_dir + r"\lh.group_gradient.npy")
    visualize_brain_surface(coords,
                            faces,
                            values,
                            title="Mean group gradient (left hemisphere)",
                            cmap="viridis",
                            threshold=0)
    
# %%
