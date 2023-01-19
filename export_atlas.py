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

    nb.save(dseg,base_name + f'_space-{atlas}_dseg.nii')
    nb.save(probseg,base_name + f'_space-{atlas}_probseg.nii')
    nb.save(Gifti,base_name + '_dseg.label.gii')
    save_lut(np.arange(len(labels)),cmap[:,0:4],labels, base_name + '.lut')

def save_lut(index,colors,labels,fname):
    """Save a set of colors and labels as a LUT file 
    Note: This should probably go into nitools 

    Args:
        index (_type_): _description_
        colors (_type_): _description_
        labels (_type_): _description_
        fname (_type_): _description_
    """
    L=pd.DataFrame({
            "key":index,
            "R":colors[:,0].round(4),
            "G":colors[:,1].round(4),
            "B":colors[:,2].round(4),
            "Name":labels})
    L.to_csv(fname,header=None,sep=' ',index=False)

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
    xs[xs<0.5]=np.nan
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

def resample_atlas(base_name):
    """ Resamples probabilistic atlas into 1mm resolution and
    SUIT space
    Note: Refactorize and make more universal 
    """
    mnisym_dir=base_dir + '/Atlases/tpl-MNI152NLin2000cSymC'
    suit_dir=base_dir + '/Atlases/tpl-SUIT'

    # Reslice to 1mm MNI and 1mm SUIT space
    print('reslicing to 1mm')
    sym3 = nb.load(mnisym_dir + f'/{base_name}_space-MNISymC3_probseg.nii')
    tmp1 = nb.load(mnisym_dir + f'/tpl-MNISymC_res-1_gmcmask.nii')
    shap = tmp1.shape+sym3.shape[3:]
    sym1 = ns.resample_from_to(sym3,(shap,tmp1.affine),3)
    print('normalizing')
    sym1,dsym1= renormalize_probseg(sym1)
    print('saving')
    nb.save(sym1,mnisym_dir + f'/{base_name}_space-MNISymC_probseg.nii')
    nb.save(dsym1,mnisym_dir + f'/{base_name}_space-MNISymC_dseg.nii')

    # Now put the image into SUIT space
    print('reslicing to SUIT')
    deform = nb.load(mnisym_dir + '/tpl-MNI152NLin2009cSymC_space-SUIT_xfm.nii')
    suit1 = nt.deform_image(sym1,deform,1)
    print('normalizing')
    suit1,dsuit1= renormalize_probseg(suit1)
    print('saving')
    nb.save(suit1,suit_dir + f'/{base_name}_space-SUIT_probseg.nii')
    nb.save(dsuit1,suit_dir + f'/{base_name}_space-SUIT_dseg.nii')
