""" Export_atlas.py
Functionality to go from fitted model to a sharable atlas (nifti/gifti)
and colormap
"""

import numpy as np
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
from scipy.linalg import block_diag
import nibabel as nb
import nibabel.processing as ns
import SUITPy as suit
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
from copy import deepcopy
import logging
import pickle
import nitools as nt
import matplotlib.pyplot as plt
import surfAnalysisPy as surf
import ProbabilisticParcellation.plot as ppp
conn_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/connectivity/maps/'
surf_dir = surf.plot._surf_dir

def renormalize_probseg(probseg, mask):
    """Renormalizes a probsegmentation file
    after resampling, so that the probabilies add up to 1

    Args:
        probseg (nifti_img):

    Returns:
        probseg_img (NiftiImage): renormalize Prob segmentation
        dseg_img (NiftiImage): desementation file
    """
    X = probseg.get_fdata()
    xs = np.sum(X, axis=3)
    X = X / np.expand_dims(xs, 3)
    maskX = mask.get_fdata()
    X[maskX == 0] = np.nan
    probseg_img = nb.Nifti1Image(X, probseg.affine)

    # explicitly detect voxels where the sum of probabilities is 0 (i.e., all NaN or zero)
    sum_prob = np.nansum(X, axis=3)
    parcel = np.argmax(np.nan_to_num(X), axis=3) + 1
    parcel[sum_prob == 0] = 0

    dseg_img = nb.Nifti1Image(parcel.astype(np.uint8), probseg.affine)
    dseg_img.set_data_dtype("uint8")
    dseg_img.header.set_intent(1002,(),"")
    probseg_img.set_data_dtype("float32")
    # probseg_img.header.set_slope_inter(1/(2**16-1),0.0)
    return probseg_img, dseg_img

def resample_probs(probs, source_space, target_space):
    """ Resamples probabilistic parcellation from MNISymC2 to a new space in 1mm resolution

    Args:
        probs (str/nifti/np.array): Probabilistic parcellation
        source_space (str): Source space (SUIT, MNISymC)
        target_space (str): Target space (MNI152NLin2009cSymC, MNI152NLin2009cAsym)

    Returns:
        nii (nifti1Image): Resampled probabilistic parcellation
        dnii (nifti1Image): Resampled winner-take-all parcellation

    """
    a, ainf = am.get_atlas(source_space, ut.atlas_dir)
    targ_dir = ut.base_dir + f"/Atlases/tpl-{target_space}"

    if isinstance(probs, str) and (probs.endswith(".nii") or probs.endswith(".nii.gz")):
        # If probs is a nifti image, load it
        nii_atlas = nb.load(probs)
    elif isinstance(probs, np.ndarray):
        # If probs is numpy array, then reshape probs to nifti
        nii_atlas = a.data_to_nifti(probs)
    elif isinstance(probs, nb.nifti1.Nifti1Image):
        nii_atlas = probs

    # Set NaNs to 0
    X = np.nan_to_num(nii_atlas.get_fdata())
    nii_atlasf = nb.Nifti1Image(X, nii_atlas.affine, nii_atlas.header)

    # Reslice to 1mm target space
    print("normalizing")
    if ainf["space"] != target_space:
        print(f"deforming from {ainf['space']} to {target_space}")
        deform = nb.load(
            targ_dir + f"/tpl-{target_space}_from-{ainf['space']}_mode-image_xfm.nii"
        )
        nii_res = nt.deform_image(nii_atlasf, deform, 1)
        # Get target space mask:
        mname = f"tpl-{target_space}_res-1_gmcmask.nii"
        nii_mask = nb.load(targ_dir + "/" + mname)
    else:
        # Make new shape
        mname = ainf["mask"]
        mname = mname.replace("res-2", "res-1")
        nii_mask = nb.load(targ_dir + "/" + mname)
        shap = nii_mask.shape + nii_atlas.shape[3:]
        nii_res = ns.resample_from_to(nii_atlasf, (shap, nii_mask.affine), 1)

    nii, dnii = renormalize_probseg(nii_res, nii_mask)
    return nii, dnii

def resample_atlas(fname, atlas="MNISymC2", target_space="MNI152NLin2009cSymC"):
    """ Resamples probabilistic atlas from MNISymC2 to a new atlas space in 1mm resolution
    Args:
        fname (str): Name of the atlas
        atlas (str/atlas): FunctionalFusion atlas (SUIT2,MNISym3, fs32k)
        target_space (str): Target space (MNI152NLin2009cSymC, MNI152NLin2009cAsym)
    """

    src_dir = ut.model_dir + "/Atlases/"
    targ_dir = ut.base_dir + f"/Atlases/tpl-{target_space}"
    source_img=src_dir + f"/{fname}_space-{atlas}_probseg.nii"

    # Resample to 1mm MNI
    nii, dnii = resample_probs(source_img, atlas, target_space)

    # Save the files for the new atlas
    print("saving")
    nb.save(nii, targ_dir + f"/atl-{fname}_space-{target_space}_probseg.nii")
    nb.save(dnii, targ_dir + f"/atl-{fname}_space-{target_space}_dseg.nii")


def prob_to_label_gii(probseg,atlas_space,cmap,label_names):
    """Generates a surface gifti file with the correct labels
    and colors from a probabilistic parcellation (nifti1Image)

    Args:
        probseg (Nifti1Image): _description_
        mapspace (str): 'SUIT' or 'MNISymC'
        cmap (nd.array): Color map including a row for 0
        label_names (list): label names (including a ??? for 0)
    """
    # Error if cmap and label_names are not the same length
    if cmap.shape[0] != len(label_names):
        raise (NameError("cmap and label_names must be the same length"))

    # Add the alpha column if it is missing
    if cmap.shape[1] ==3:
        cmap = np.concatenate((cmap,np.ones((cmap.shape[0],1))),axis=1)

    # Figure out correct mapping space
    if atlas_space.startswith("SUIT"):
        map_space = "SUIT"
    elif atlas_space.startswith("MNISymC"):
        map_space = "MNISymC"
    elif atlas_space == "MNI152NLin2009cSymC":
        map_space = "MNISymC"
    elif atlas_space == "MNI152NLin6AsymC":
        map_space = "FSL"
    else:
        raise (NameError("Unknown atlas space"))

    # Plotting label
    surf_data = suit.flatmap.vol_to_surf(probseg,
                                         stats="nanmean",
                                         space=map_space)
    surf_parcel = np.argmax(surf_data, axis=1) + 1
    gifti = nt.make_label_gifti(
        surf_parcel.reshape(-1, 1),
        anatomical_struct="Cerebellum",
        labels=np.arange(probseg.shape[3] + 1),
        label_names=label_names,
        label_RGBA=cmap)
    return gifti


def export_map(data, atlas, cmap, label_names, base_name):
    """Exports a marginal probability of a arrangement model to a Nifti (probseg), Nifti (dseg), Gifti, and lut-file.

    Args:
        data (probabilities): Marginal probabilities of the arrangement model
        atlas (str/atlas): FunctionalFusion atlas (SUIT2,MNISym3, fs32k)
        cmap (ListedColormap): Colormap
        labels (list): List of labels for fields
        base_name (_type_): File directory + basename for atlas
    """
    # Error if cmap and label_names are not the same length
    if cmap.shape[0] != len(label_names):
        raise (NameError("cmap and label_names must be the same length"))
    # Transform cmap into numpy array
    if not isinstance(cmap, np.ndarray):
        cmap = cmap(np.arange(cmap.N))

    suit_atlas, _ = am.get_atlas(atlas, ut.base_dir + "/Atlases")
    probseg = suit_atlas.data_to_nifti(data)
    parcel = np.argmax(data, axis=0) + 1
    parcel = parcel.astype(np.int8)
    dseg = suit_atlas.data_to_nifti(parcel)

    Gifti = prob_to_label_gii(probseg, atlas, cmap, label_names)

    nb.save(dseg, base_name + f"_dseg.nii")
    nb.save(probseg, base_name + f"_probseg.nii")
    nb.save(Gifti, base_name + "_dseg.label.gii")
    # nt.save_lut(base_name + ".lut", np.arange(len(labels)), cmap[:, 0:4], labels)
    print(f"Exported {base_name}.")


def get_spatial_compartments():
    """ Returns the spatial compartments for the 32 region atlas

        Spatial subdivisions are:
            Superior (lobule I - Crus I inclusive)
            Dorsal inferior (Crus II - VIIIb)
            Ventral inferior (lobule IX - lobule X)
            Vermal inferior sections (vermis VII - vermis X)

        Returns:
            sp_ext (list): List of strings for subregion names
            comp (list): List of lists of anatomical labels
    """
    sp_ext = ['s','i','t','v']
    comp = [[1,2,3,4,5,6,7,8,10],
            [11,13,14,16,17,19,20,22],
            [23,25,26,28],
            [9,12,15,18,21,24,27]]

    return sp_ext,comp

def divide_map(prob, anat, sp_ext, comp, labels=None, colors=None):
    """Divides a probabilistic map into subregions
    Args:
        prob (ndarray): Probabilistic map
        anat (ndarray): Anatomical map
        sp_ext (list): List of strings for subregion names
        comp (list): List of lists of anatomical labels
        labels (list): List of labels for fields
        colors (ndarray): Color map including a row for 0
    Returns:
        prob_new (ndarray): New probabilistic map
        indx_new (ndarray): New index
        colors_new (ndarray): New color map
        labels_new (list): New list of labels
    """

    nx,ny,nz,K = prob.shape
    prob_new = np.zeros((nx,ny,nz,K*4))
    indx_new = np.arange(K*4+1)
    colors_new = np.zeros((K*4+1,3))
    labels_new = ['0']

    # Loop over all regions and subdivide them
    for k in range(K):
        for i,(s,compartment) in enumerate(zip(sp_ext,comp)):
            inew = k*4+i
            prob_new[:,:,:,inew] = prob[:,:,:,k]*(np.isin(anat,compartment))
            if labels is not None:
                labels_new.append(labels[k+1] + s)
            if colors is not None:
                colors_new[inew+1,:] = colors[k+1,:]

    parcel = np.argmax(np.nan_to_num(prob_new), axis=3)
    sumOfProb = np.nansum(prob_new,axis=3)
    parcel[sumOfProb==0]=0
    return prob_new,parcel,indx_new,colors_new,labels_new

def subdivde_atlas_spatial(fname,atlas,outname):
    """ Subdivides the 32 region atlas into s,i,t,v
    It performs this on the already resampled atlases in the FunctionFusion/atlas directory.
        s: superior (lobule I-crusI)
        i: inferior (crus II - VIIIb)
        t: tertiary (IX/X)
        v: vermal (inferior vermis)
    Args:
        fname (str): Name of map (e.g. Nettekoven32)
        atlas (str): Atlas name (e.g. MNI152NLin2009cSymC)
    """
    # Get spatial compartments
    sp_ext,comp = get_spatial_compartments()

    # Get anatomical image
    tpl_dir = ut.atlas_dir + f"/tpl-{atlas}/"
    anat_atlas = tpl_dir + f"atl-Anatom_space-{atlas}_dseg.nii"
    anat_img = nb.load(anat_atlas)
    anat = anat_img.get_fdata()

    # Get atlas and set NaNs to 0
    base_name = f"atl-{fname}_space-{atlas}"
    prob_atlas = tpl_dir + base_name + '_probseg.nii'
    lutfile = tpl_dir + f"atl-{fname}.lut"
    prob_img = nb.load(prob_atlas)
    prob = prob_img.get_fdata()

    # Load Lut file and make new version of it
    indx,colors,labels = nt.read_lut(lutfile)

    # Divide the map
    prob_new,parcel,indx_new,colors_new,labels_new = divide_map(prob,anat,sp_ext,comp,labels,colors)

    # Save new atlas
    pseg_img = nb.Nifti1Image(prob_new, prob_img.affine)
    dseg_img = nb.Nifti1Image(parcel.astype(np.uint8),prob_img.affine)
    gifti = prob_to_label_gii(pseg_img, atlas, colors_new, labels_new)

    out = f"atl-{outname}_space-{atlas}"

    #Save the files for the new atlas
    nb.save(dseg_img, tpl_dir + f"atl-{outname}_space-{atlas}_dseg.nii")
    nb.save(pseg_img, tpl_dir + f"atl-{outname}_space-{atlas}_probseg.nii")
    nb.save(gifti, tpl_dir + f"atl-{outname}_dseg.label.gii")
    nt.save_lut(tpl_dir + f"atl-{outname}.lut", indx_new, colors_new, labels_new)


def export_conn_summary():
    """Exports the connectivity profiles of all parcels"""
    # load labels
    index, cmap, labels = nt.read_lut(
            ut.export_dir
            + "NettekovenSym32.lut")

    for parcel in labels[1:len(labels)-1//2]:
        parcel=parcel[:2]
        ppp.plot_parcel_summary(parcel=parcel,atlas='NettekovenSym32',space='MNISymC2')
        # Make layout tighter
        plt.tight_layout()
        plt.savefig(f'{ut.figure_dir}/parcel_summary_{parcel}.png')

    pass


def export_all_probmaps():
    index, cmap, labels = nt.read_lut(
            ut.export_dir
            + "NettekovenSym32.lut"
        )

    for parcel in labels[1:len(labels)-1//2]:
        parcel=parcel[:2]

        plt.figure(figsize=(8, 8))
        ppp.plot_parcel_prob(parcel,'NettekovenSym32',space='MNISymC2',backgroundcolor='w',bordercolor='k')
        plt.savefig(ut.figure_dir + f'Prob_{parcel}.png',bbox_inches='tight')

def save_cortex_cifti(fname):
    """Exports a cortical model as a surface-based CIFTI label file.
    Args:
        fname (str): model name
    """
    info, model = ut.load_batch_best(fname)
    Prop = model.marginal_prob()
    par = pt.argmax(Prop, dim=0) + 1
    atlas, _ = am.get_atlas("fs32k", ut.atlas_dir)
    img = nt.make_label_cifti(par.numpy(), atlas.get_brain_model_axis())
    nb.save(img, ut.model_dir + f"/Models/{fname}.dlabel.nii")



def colour_parcel(mname, sym=False, plot=True, labels=None, clusters=None, weighting=None, gamma=0):
    """
    Colours the parcellation of a model.

    Args:
    - mname (str): Path of the model to be analyzed.
    - sym (bool): Whether to generate similarity in a symmetric fashion. Defaults to True.
    - plot (bool): Whether or not to generate plots. Defaults to True.
    - labels (ndarray): Labels for the parcels if they have already been generated. Defaults to None.
    - clusters (ndarray): Distorts color towards cluster mean.
    - weighting (str): Type of weighting to use for calculating parcel similarity. Defaults to None.
    - gamma (float): The gamma value used for the colormap.

    Returns:
    - Prob (ndarray): The winner-take-all probabilities for each region.
    - parcel (ndarray): The parcel label for each region.
    - atlas (object): The atlas object used for the parcellation.
    - labels (ndarray): The labels for the clusters generated by clustering.
    - cmap (object): The colormap generated for the parcellation.
    """

    # Get model and atlas.
    fileparts = mname.split("/")
    split_mn = fileparts[-1].split("_")
    info, model = ut.load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.arrange.marginal_prob())
    parcel = Prob.argmax(axis=0) + 1

    # Make a colormap.
    w_cos_sim, _, _ = cl.parcel_similarity(model, plot=False, sym=sym)
    W = sc.calc_mds(w_cos_sim, center=True)
    if sym:
        W = np.concatenate([W, W])
    # Define color anchors
    m, regions, colors = sc.get_target_points(atlas, parcel)
    cmap = sc.colormap_mds(
        W, target=(m, regions, colors), clusters=clusters, gamma=gamma
    )
    sc.plot_colorspace(cmap(np.arange(model.K)))

    plt.figure(figsize=(5, 10))
    cl.plot_parcel_size(Prob, cmap, labels, wta=True)

    # Plot the parcellation
    if plot:
        ax = ut.plot_data_flat(
            Prob, atlas.name, cmap=cmap, dtype="prob", labels=labels, render="plotly"
        )
        ax.show()

    return Prob, parcel, atlas, labels, cmap

def export_model_map():
    info,M = ut.load_batch_best('Models_03/asym_Md_space-MNISymC3_K-17')
    Data = M.arrange.marginal_prob().cpu().numpy()
    suit_atlas, _ = am.get_atlas('MNISymC3', base_dir + "/Atlases")
    probseg = suit_atlas.data_to_nifti(Data)
    nb.save(probseg, "asym_Md_space-MNISymC3_K-17_probseg.nii")


if __name__ == "__main__":

    info,M = ut.load_batch_best('Models_03/asym_Md_space-MNISymC3_K-17')
    Data = M.arrange.marginal_prob()
    pass