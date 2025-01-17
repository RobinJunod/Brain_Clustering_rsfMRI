# Full Pipeline Instructions

Below are the steps required to run the full pipeline correctly.

---
## 1. Steps to Generate Important Surface Files

### 1.1. Minimal fMRI Preprocessing
1. Perform your minimal fMRI preprocessing as required.

### 1.2. FreeSurfer `recon-all`
2. Run:
  ```bash
  recon-all -s <subject_name> -all
  ```
---

## 2. Modify the `process_reconall.sh` Script

1. Update your subject directory paths in `process_reconall.sh`.  
2. Run the script to process all of the surface meshes.

---

### 2.1 Details on the `process_reconall.sh` (infos) 
Freesurfer and wb_command (from the HCP project) must be installed.

---
#### (1) Convert to .gii

```bash
mris_convert lh.white lh.white.surf.gii```
mris_convert lh.pial.T1 lh.pial.surf.gii
mris_convert rh.white rh.white.surf.gii
mris_convert rh.pial.T1 rh.pial.surf.gii
```

#### (2) Create the midthickness surface
```bash
wb_command -surface-average lh.midthickness.surf.gii -surf lh.white.surf.gii -surf lh.pial.surf.gii
wb_command -surface-average rh.midthickness.surf.gii -surf rh.white.surf.gii -surf rh.pial.surf.gii
```

#### (3) Downsample the surface mesh to ~32k vertices
```bash
mris_remesh -i rh.midthickness.surf.gii -o rh.midthickness.32k.surf.gii --nvert 32492
mris_remesh -i lh.midthickness.surf.gii -o lh.midthickness.32k.surf.gii --nvert 32492
mris_remesh -i lh.pial.surf.gii -o lh.pial.32k.surf.gii --nvert 32492
mris_remesh -i rh.pial.surf.gii -o rh.pial.32k.surf.gii --nvert 32492
mris_remesh -i lh.white.surf.gii -o lh.white.32k.surf.gii --nvert 32492
mris_remesh -i rh.white.surf.gii -o rh.white.32k.surf.gii --nvert 32492
```


#### (4) Inflate the midthickness
```bash
mris_inflate -n 3 -dist 0.1 -no-save-sulc lh.midthickness.32k.surf.gii lh.midthickness.inflated.32k.surf.gii

```

```bash
mris_inflate -n 3 -dist 0.1 -no-save-sulc rh.midthickness.32k.surf.gii rh.midthickness.inflated.32k.surf.gii
```


## Imporatn for group analysis
cd /mnt/d/DATA_min_preproc/dataset_study1
export SUBJECTS_DIR=/home/rjunod/freesurfer/study1

### Project fmri data into subject surface 
```bash
# Create a surface projection of the functional data IN NATIVE SPACE
mri_vol2surf \
  --mov sub-02/func/wsraPPS-FACE_S02_005_Rest.nii.gz \
  --regheader sub02_freesurfer \
  --hemi lh \
  --surf white \
  --projfrac 0.5 \
  --interp trilinear \
  --o sub-02/func/sub02_lh.func.mgh # outputs it in the folder the command has benn executed

mri_vol2surf \
  --mov sub-02/func/wsraPPS-FACE_S02_005_Rest.nii.gz \
  --regheader sub02_freesurfer \
  --hemi rh \
  --surf white \
  --projfrac 0.5 \
  --interp trilinear \
  --o sub-02/func/sub02_rh.func.mgh
```

### Project the surface time series into fsaverage6

```bash
mri_surf2surf \
  --srcsubject sub02_freesurfer \
  --trgsubject fsaverage6 \
  --hemi lh \
  --sval sub-02/func/sub02_lh.func.mgh \
  --tval sub-02/func/sub02_lh.func.fsaverage6.mgh \
  --surfreg sphere.reg

mri_surf2surf \
  --srcsubject sub02_freesurfer \
  --trgsubject fsaverage6 \
  --hemi rh \
  --sval sub-02/func/sub02_rh.func.mgh \
  --tval sub-02/func/sub02_rh.func.fsaverage6.mgh \
  --surfreg sphere.reg
```

