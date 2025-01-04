HERE is the instructoon to run the full pipeline correctly



Modify the 'process_reconall.sh' with your subj dir paths.



below are the steps to make with freesurfer and wb_command in order to get all of the important surfaces files:

- 1 min preprocessing fmri
- 2 freesurfer recon -all

- 3 convert to .gii
mris_convert lh.white lh.white.surf.gii
mris_convert lh.pial.T1 lh.pial.surf.gii
mris_convert rh.white rh.white.surf.gii
mris_convert rh.pial.T1 rh.pial.surf.gii

- 4 create the midthickness surface
wb_command -surface-average lh.midthickness.surf.gii -surf lh.white.surf.gii -surf lh.pial.surf.gii
wb_command -surface-average rh.midthickness.surf.gii -surf rh.white.surf.gii -surf rh.pial.surf.gii

- 5 Resample to the pial (32k)
mris_remesh -i rh.midthickness.surf.gii -o rh.midthickness.32k.surf.gii --nvert 32492
mris_remesh -i lh.midthickness.surf.gii -o lh.midthickness.32k.surf.gii --nvert 32492

mris_remesh -i lh.pial.surf.gii -o lh.pial.32k.surf.gii --nvert 32492
mris_remesh -i rh.pial.surf.gii -o rh.pial.32k.surf.gii --nvert 32492

mris_remesh -i lh.white.surf.gii -o lh.white.32k.surf.gii --nvert 32492
mris_remesh -i rh.white.surf.gii -o rh.white.32k.surf.gii --nvert 32492

- 6 inflate the midthickness
mris_inflate \
  -n 3 \
  -dist 0.1 \
  -no-save-sulc \
  lh.midthickness.32k.surf.gii \
  lh.midthickness.inflated.32k.surf.gii

mris_inflate \
  -n 3 \
  -dist 0.1 \
  -no-save-sulc \
  rh.midthickness.32k.surf.gii \
  rh.midthickness.inflated.32k.surf.gii