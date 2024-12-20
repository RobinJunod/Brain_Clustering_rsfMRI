#%%
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn as nl


# Step 1: Load the surface data
# Load the surface data

# read gifti file
rh_func_path = r"D:\DATA_min_preproc\dataset_study2\sub-01\func\rh.func.native.gii"

# Load the .gii file
gii_data = nib.load(rh_func_path)

# Access the data array
fmri_data = np.array([darray.data for darray in gii_data.darrays]).T  # Shape: (n_vertices, n_timepoints)
print(f"Data shape: {fmri_data.shape}")  # Should be (n_vertices, n_timepoints)

