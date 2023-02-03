""" Export_atlas.py
Functionality to go from fitted model to a sharable atlas (nifti/gifti) 
and colormap 
"""

import numpy as np
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
from scipy.linalg import block_diag
import nibabel as nb
import nibabel.processing as ns
import SUITPy as suit
from ProbabilisticParcellation.util import *
from copy import deepcopy

def save_cortex_cifti(fname):
    """Exports a cortical model as a surface-based CIFTI label file.
    Args:
        fname (str): model name
    """
    info, model = load_batch_best(fname)
    Prop = model.marginal_prob()
    par = pt.argmax(Prop, dim=0) + 1
    atlas,_ = am.get_atlas('fs32k', atlas_dir)
    img = nt.make_label_cifti(par.numpy(), atlas.get_brain_model_axis())
    nb.save(img, model_dir + f'/Models/{fname}.dlabel.nii')

def export_map(data,atlas,cmap,labels,base_name):
    """Exports a new atlas map as a Nifti (probseg), Nifti (desg), Gifti, and lut-file.

    Args:
        data (probabilities): Marginal probabilities of the arrangement model 
        atlas (str/atlas): FunctionalFusion atlas (SUIT2,MNISym3)
        cmap (ListedColormap): Colormap
        labels (list): List of labels for fields
        base_name (_type_): File directory + basename for atlas
    """
    # Transform cmap into numpy array
    if not isinstance(cmap,np.ndarray):
        cmap = cmap(np.arange(cmap.N))


    suit_atlas, _ = am.get_atlas(atlas,base_dir + '/Atlases')
    probseg = suit_atlas.data_to_nifti(data)
    parcel = np.argmax(data,axis=0)+1
    dseg = suit_atlas.data_to_nifti(parcel)

    # Figure out correct mapping space
    if atlas[0:4]=='SUIT':
        map_space='SUIT'
    elif atlas[0:7]=='MNISymC':
        map_space='MNISymC'
    else:
        raise(NameError('Unknown atlas space'))

    # Plotting label
    surf_data = suit.flatmap.vol_to_surf(probseg, stats='nanmean',
            space=map_space)
    surf_parcel = np.argmax(surf_data,axis=1)+1
    Gifti = nt.make_label_gifti(surf_parcel.reshape(-1,1),
                anatomical_struct='Cerebellum',
                labels = np.arange(surf_parcel.max()+1),
                label_names=labels,
                label_RGBA = cmap)

    nb.save(dseg,base_name + f'_dseg.nii')
    nb.save(probseg,base_name + f'_probseg.nii')
    nb.save(Gifti,base_name + '_dseg.label.gii')
    nt.save_lut(base_name,np.arange(len(labels)),cmap[:,0:4],labels)
    print(f'Exported {base_name}.')

def renormalize_probseg(probseg):
    """ Renormalizes a probsegmentation file 
    after resampling, so that the probabilies add up to 1 

    Args:
        probseg (nifti_img): 

    Returns:
        probseg_img (NiftiImage): renormalize Prob segmentation 
        dseg_img (NiftiImage): desementation file 
    """
    X = probseg.get_fdata()
    xs = np.sum(X,axis=3)
    xs[xs<0.3]=np.nan
    X = X/np.expand_dims(xs,3)
    X[np.isnan(X)]=0
    probseg_img = nb.Nifti1Image(X,probseg.affine)
    parcel = np.argmax(X,axis=3)+1
    parcel[np.isnan(xs)]=0
    dseg_img = nb.Nifti1Image(parcel.astype(np.int8),probseg.affine)
    dseg_img.set_data_dtype('int8')
    # dseg_img.header.set_intent(1002,(),"")
    probseg_img.set_data_dtype('float32')
    # probseg_img.header.set_slope_inter(1/(2**16-1),0.0)
    return probseg_img,dseg_img

def resample_atlas(fname,
        atlas='MNISymC2',
        target_space='MNI152NLin2009cSymC'):
    """ Resamples probabilistic atlas from MNISymC2 to a new atlas space 1mm resolution
    """
    a,ainf = am.get_atlas(atlas,atlas_dir)
    src_dir= model_dir + '/Atlases/'
    targ_dir= base_dir + f'/Atlases/tpl-{target_space}'
    srcs_dir = base_dir + '/Atlases/' + ainf['dir']
    nii_atlas = nb.load(src_dir + f'/{fname}_probseg.nii')
    # Reslice to 1mm MNI
    if ainf['space'] != target_space:
        print(f"deforming from {ainf['space']} to {target_space}")
        deform = nb.load(srcs_dir + f"/tpl-{ainf['space']}_space-{target_space}_xfm.nii")
        nii_res = nt.deform_image(nii_atlas,deform,1)
    else:
        mname = ainf['mask']
        mname = mname.replace('res-2','res-1')
        nii_mask = nb.load(targ_dir + '/' + mname)
        # Make new shape 
        shap = nii_mask.shape+nii_atlas.shape[3:]
        nii_res = ns.resample_from_to(nii_atlas,(shap,nii_mask.affine),1)
    print('normalizing')
    nii,dnii= renormalize_probseg(nii_res)
    print('saving')
    nb.save(nii,targ_dir + f'/atl-{fname}_space-{target_space}_probseg.nii')
    nb.save(dnii,targ_dir + f'/atl-{fname}_space-{target_space}_dseg.nii')

