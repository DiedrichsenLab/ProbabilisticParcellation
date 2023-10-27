#!/bin/bash
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Extract Crus I and II from SUIT space
lobules=/Applications/spm12/toolbox/suit/atlasesSUIT/Lobules-SUIT.nii

for i in 10 13 8 11; do
    fslmaths ${lobules} -thr $i -uthr $i -bin lobules_$i
done

# Rename lobules 13 and 11 to Crus I left and right
mv lobules_13.nii.gz CrusI_left.nii.gz
mv lobules_11.nii.gz CrusI_right.nii.gz

# Rename lobules 10 and 8 to Crus II left and right
mv lobules_10.nii.gz CrusII_left.nii.gz
mv lobules_8.nii.gz CrusII_right.nii.gz

# Dilate Crus I on both sides 1 voxel
for hem in left right; do
    fslmaths CrusI_${hem}.nii.gz -kernel 3D -dilM CrusI_${hem}_dilated
done

# Add dilated Crus I to Crus II
for hem in left right; do
    fslmaths CrusI_${hem}_dilated -add CrusII_${hem} -thr 2 -bin horizontal_fissure_${hem}
done

# Add left and right fissures into one image
fslmaths horizontal_fissure_left -add horizontal_fissure_right -bin horizontal_fissure
# Unzip nifti file
gunzip horizontal_fissure.nii.gz
# Copy to tpl-SUIT folder
scp horizontal_fissure.nii  /Volumes/diedrichsen_data\$/data/FunctionalFusion/Atlases/tpl-SUIT/atl-SUIT_fissures.nii