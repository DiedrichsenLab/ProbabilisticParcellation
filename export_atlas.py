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

def subdivde_atlas_spatial(fname,atlas):
    """ Subdivides the 32 region atlas into s,i,t,v
        s: superior (lobule I-crusI)
        i: inferior (crus II - VIIIb)
        t: tertiary (IX/X)
        v: vermal (inferior vermis)
    Args:
        atlas (str): Atlas name
    """
    sp_ext = ['s','i','t','v']
    comp = [[1,2,3,4,5,6,7,8,10],
            [11,13,14,16,17,19,20,22], 
            [23,25,26,28],
            [9,12,15,18,21,24,27]]
    a, ainf = am.get_atlas(atlas, ut.atlas_dir)
    src_dir = ut.model_dir + "/Atlases/"

    # Load and set NaNs to 0
    base_name = f"{fname}_space-{atlas}"
    prob_atlas = src_dir + base_name + '_probseg.nii' 
    anat_atlas = ut.atlas_dir + f"/tpl-{ainf['space']}/atl-Anatom_space-{ainf['space']}_dseg.nii"
    lutfile = src_dir + f"{fname}.lut"
    prob = a.read_data(prob_atlas,0)
    anat = a.read_data(anat_atlas,0)
    P,K = prob.shape
    prob_new = np.zeros((P,K*4))

    # Load Lut file and make new version of it 
    indx,colors,labels = nt.read_lut(lutfile)
    indx_new = np.arange(K*4+1)
    colors_new = np.zeros((K*4+1,3))
    labels_new = ['0'] 

    # Loop over all regions and subdivide them
    for k in range(K): 
        for i,(s,compartment) in enumerate(zip(sp_ext,comp)):
            inew = k*4+i
            prob_new[:,inew] = prob[:,k]*(np.isin(anat,compartment))
            labels_new.append(labels[k+1] + s)
            colors_new[inew+1,:] = colors[k+1,:]
    
    # Save new atlas
    parcel = np.argmax(prob_new,axis=1)+1
    parcel[prob_new.sum(axis=1)==0]=0
    probseg = a.data_to_nifti(prob_new.T)
    parcel = parcel.astype(np.int16)
    dseg = a.data_to_nifti(parcel)

    out_name = fname + 'sp'
    nb.save(dseg, src_dir + out_name + f"_space-{atlas}_dseg.nii")
    nb.save(probseg, src_dir + out_name + f"_space-{atlas}_probseg.nii")
    nt.save_lut(src_dir + out_name + ".lut", indx_new, colors_new, labels_new)

        
def export_conn_summary():
    """Exports the connectivity profiles of all parcels"""
    # load labels
    index, cmap, labels = nt.read_lut(
            ut.export_dir
            + "NettekovenSym32.lut"
        )

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


def export_map(data, atlas, cmap, labels, base_name):
    """Exports a marginal probability of a arrangement model to a Nifti (probseg), Nifti (dseg), Gifti, and lut-file.

    Args:
        data (probabilities): Marginal probabilities of the arrangement model
        atlas (str/atlas): FunctionalFusion atlas (SUIT2,MNISym3, fs32k)
        cmap (ListedColormap): Colormap
        labels (list): List of labels for fields
        base_name (_type_): File directory + basename for atlas
    """
    # Transform cmap into numpy array
    if not isinstance(cmap, np.ndarray):
        cmap = cmap(np.arange(cmap.N))

    suit_atlas, _ = am.get_atlas(atlas, ut.base_dir + "/Atlases")
    probseg = suit_atlas.data_to_nifti(data)
    parcel = np.argmax(data, axis=0) + 1
    parcel = parcel.astype(np.int8)
    dseg = suit_atlas.data_to_nifti(parcel)

    # Figure out correct mapping space
    if atlas[0:4] == "SUIT":
        map_space = "SUIT"
    elif atlas[0:7] == "MNISymC":
        map_space = "MNISymC"
    else:
        raise (NameError("Unknown atlas space"))

    # Plotting label
    surf_data = suit.flatmap.vol_to_surf(probseg, stats="nanmean", space=map_space)
    surf_parcel = np.argmax(surf_data, axis=1) + 1
    Gifti = nt.make_label_gifti(
        surf_parcel.reshape(-1, 1),
        anatomical_struct="Cerebellum",
        labels=np.arange(surf_parcel.max() + 1),
        label_names=labels,
        label_RGBA=cmap,
    )

    nb.save(dseg, base_name + f"_dseg.nii")
    nb.save(probseg, base_name + f"_probseg.nii")
    nb.save(Gifti, base_name + "_dseg.label.gii")
    # nt.save_lut(base_name + ".lut", np.arange(len(labels)), cmap[:, 0:4], labels)
    print(f"Exported {base_name}.")


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
    parcel = np.argmax(X, axis=3) + 1
    parcel[maskX == 0] = 0
    dseg_img = nb.Nifti1Image(parcel.astype(np.uint8), probseg.affine)
    dseg_img.set_data_dtype("uint8")
    dseg_img.header.set_intent(1002,(),"")
    probseg_img.set_data_dtype("float32")
    # probseg_img.header.set_slope_inter(1/(2**16-1),0.0)
    return probseg_img, dseg_img


def resample_atlas(fname, atlas="MNISymC2", target_space="MNI152NLin2009cSymC"):
    """ Resamples probabilistic atlas from MNISymC2 to a new atlas space in 1mm resolution
    Args:
        fname (str): Name of the atlas
        atlas (str/atlas): FunctionalFusion atlas (SUIT2,MNISym3, fs32k)
        target_space (str): Target space (MNI152NLin2009cSymC, MNI152NLin2009cAsym)
    """
    a, ainf = am.get_atlas(atlas, ut.atlas_dir)
    src_dir = ut.model_dir + "/Atlases/"
    targ_dir = ut.base_dir + f"/Atlases/tpl-{target_space}"
    # Load and set NaNs to 0
    nii_atlas = nb.load(src_dir + f"/{fname}_space-{atlas}_probseg.nii")
    X = np.nan_to_num(nii_atlas.get_fdata())
    nii_atlasf = nb.Nifti1Image(X, nii_atlas.affine, nii_atlas.header)
    # Reslice to 1mm MNI
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
    print("saving")
    nb.save(nii, targ_dir + f"/atl-{fname}_space-{target_space}_probseg.nii")
    nb.save(dnii, targ_dir + f"/atl-{fname}_space-{target_space}_dseg.nii")


def reorder_model(
    mname,
    sym=True,
    mname_new=None,
    assignment="mixed_assignment_68_16.csv",
    original_idx="parcel_orig_idx",
    save_model=False,
):
    """
    Reorders a saved parcellation model according to fixed order assignment.

    Args:
        mname (str): The name of the saved model to be reordered.
        sym (bool): If True, reorders the model assuming symmetrical model.
        mname_new (str): The name of the reordered model. If None, the name will be the same as the original model with '_reordered' appended.
        assignment (str): The name of the CSV file containing the order assignment.
        original_idx (str): The name of the column in the assignment CSV file containing the original parcel indices.
        save_model (bool): If True, saves the reordered model.

    Returns:
        new_model (object): The reordered model object.

    """
    # Get model and atlas.
    fileparts = mname.split("/")
    split_mn = fileparts[-1].split("_")
    info, model = ut.load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Get assignment
    assignment = pd.read_csv(f"{ut.model_dir}/Atlases/{assignment}")

    order_arrange = assignment[original_idx].values

    # Reorder the model
    new_model = deepcopy(model)
    if new_model.arrange.logpi.shape[0] == order_arrange.shape[0]:
        new_model.arrange.logpi = model.arrange.logpi[order_arrange]
    elif new_model.arrange.logpi.shape[0] * 2 == order_arrange.shape[0]:
        new_model.arrange.logpi = model.arrange.logpi[
            order_arrange[: len(order_arrange) // 2]
        ]
    elif new_model.arrange.logpi.shape[0] == np.unique(order_arrange).shape[0]:
        # Make order_arrange unique list of indices in the same order (necessary for re-ordering already merged models)
        order_arrange = order_arrange[
            np.sort(np.unique(order_arrange, return_index=True)[1])
        ]
        new_model.arrange.logpi = model.arrange.logpi[order_arrange]
    else:
        raise ValueError(
            "The number of parcels in the model does not match the number of parcels in the assignment."
        )

    order_emission = np.concatenate([order_arrange, order_arrange + len(order_arrange)])
    if not sym:
        order_arrange = order_emission

    for e, em in enumerate(new_model.emissions):
        new_model.emissions[e].V = em.V[:, order_emission]

    # Info
    new_info = deepcopy(info)
    new_info["ordered_by"] = original_idx
    new_info = new_info.to_frame().T

    # Save the model
    if save_model:
        if mname_new is None:
            mname_new = mname + "_reordered"
        # save new model
        with open(f"{ut.model_dir}/Models/{mname_new}.pickle", "wb") as file:
            pickle.dump([new_model], file)

        # save new info
        new_info.to_csv(f"{ut.model_dir}/Models/{mname_new}.tsv", sep="\t", index=False)

        print(
            f"Done. Saved reordered model as: \n\t{mname_new} \nOutput folder: \n\t{ut.model_dir}/Models/ \n\n"
        )

    return new_model


def colour_parcel(
    mname, sym=False, plot=True, labels=None, clusters=None, weighting=None, gamma=0
):
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


if __name__ == "__main__":
    # export_conn_summary()
    # export_all_probmaps()
    subdivde_atlas_spatial(fname='NettekovenSym32',atlas='MNISymC2') 