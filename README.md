# 🧠 RS-fMRI Parcellation Toolkit  
*Subject and Group Level Brain Clustering — from raw scans fmri to high-resolution surface & volume atlases*

> **Master Thesis · CHUV / EPFL ― “MYSPACE” project**  
> **Author :** Robin Junod  ·  **Supervisors :** Dr. Michel Akselrod 

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/) 
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) 

---

## ✨ What’s inside?

| Stage | Purpose | Key scripts / notebooks |
|-------|---------|-------------------------|
| **0 · Pre-processing** | Convert raw DICOM → BIDS, motion-correct, normalise, smooth | *external*: **SPM**, **CONN**, **Freesurfer** |
| **1 · Surface pipeline** | Subject-level parcellation on *fsaverage* cortical mesh | `process_reconall.sh`, `vol2fsaverage.sh`, `surface_parcellation.ipynb` |
| **2 · Volume pipeline** | 3-D clustering straight in native/standard space | `parcellation_volume/volume_parcellation.py`, `volume_demo.ipynb` |
| **3 · Visualisation & QC** | Quick-look plots, interactive QC HTML | `viz/plot_parcels.py`, `qc_report.ipynb` |

👉 **Result:** a personalised atlas of cortical (surface) and sub-cortical (volume) parcels ready for connectivity or graph-theory analysis.

---
## Preprocessing

- USE [SPM](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) for the minimal preprocessing steps (mo corr, time slicing, etc.)
- USE [CONN](https://web.conn-toolbox.org/) toolbox for noise Artefact reduction (recommended).

## 🏁 Quick Start (5 steps)

```bash
# 1 ▸ clone
git clone https://github.com/RobinJunod/Brain_Clustering_rsfMRI.git
cd Brain_Clustering_rsfMRI

# 2 ▸ create identical environment
conda env create -f environment.yml
conda activate rsfmri_parc

# 3 ▸ put your data in BIDS format
└── sub-01/
    ├── anat/sub-01_T1w.nii.gz
    └── func/sub-01_task-rest_bold.nii.gz

# 4 ▸ run surface pipeline (example)
bash scripts/process_reconall.sh sub-01
bash scripts/vol2fsaverage.sh sub-01
jupyter notebook surface_parcellation.ipynb     # tweak & run

# 5 ▸ run volume pipeline
python parcellation_volume/volume_parcellation.py \
       --bold sub-01_task-rest_bold.nii.gz \
       --brainmask sub-01_brainmask.nii.gz \
       --roi sub-01_S1roi.nii.gz
```

## Detailed Results
[Take a look at the PDF report](results/EPFL_Master_Thesis.pdf)


