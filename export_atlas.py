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
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
from copy import deepcopy
import logging
import pickle


def save_cortex_cifti(fname):
    """Exports a cortical model as a surface-based CIFTI label file.
    Args:
        fname (str): model name
    """
    info, model = ut.load_batch_best(fname)
    Prop = model.marginal_prob()
    par = pt.argmax(Prop, dim=0) + 1
    atlas, _ = am.get_atlas('fs32k', ut.atlas_dir)
    img = nt.make_label_cifti(par.numpy(), atlas.get_brain_model_axis())
    nb.save(img, ut.model_dir + f'/Models/{fname}.dlabel.nii')


def export_map(data, atlas, cmap, labels, base_name):
    """Exports a new atlas map as a Nifti (probseg), Nifti (desg), Gifti, and lut-file.

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

    suit_atlas, _ = am.get_atlas(atlas, ut.base_dir + '/Atlases')
    probseg = suit_atlas.data_to_nifti(data)
    parcel = np.argmax(data, axis=0) + 1
    dseg = suit_atlas.data_to_nifti(parcel)

    # Figure out correct mapping space
    if atlas[0:4] == 'SUIT':
        map_space = 'SUIT'
    elif atlas[0:7] == 'MNISymC':
        map_space = 'MNISymC'
    else:
        raise (NameError('Unknown atlas space'))

    # Plotting label
    surf_data = suit.flatmap.vol_to_surf(probseg, stats='nanmean',
                                         space=map_space)
    surf_parcel = np.argmax(surf_data, axis=1) + 1
    Gifti = nt.make_label_gifti(surf_parcel.reshape(-1, 1),
                                anatomical_struct='Cerebellum',
                                labels=np.arange(surf_parcel.max() + 1),
                                label_names=labels,
                                label_RGBA=cmap)

    nb.save(dseg, base_name + f'_dseg.nii')
    nb.save(probseg, base_name + f'_probseg.nii')
    nb.save(Gifti, base_name + '_dseg.label.gii')
    nt.save_lut(base_name + '.lut',
                np.arange(len(labels)), cmap[:, 0:4], labels)
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
    xs = np.sum(X, axis=3)
    xs[xs < 0.3] = np.nan
    X = X / np.expand_dims(xs, 3)
    X[np.isnan(X)] = 0
    probseg_img = nb.Nifti1Image(X, probseg.affine)
    parcel = np.argmax(X, axis=3) + 1
    parcel[np.isnan(xs)] = 0
    dseg_img = nb.Nifti1Image(parcel.astype(np.int8), probseg.affine)
    dseg_img.set_data_dtype('int8')
    # dseg_img.header.set_intent(1002,(),"")
    probseg_img.set_data_dtype('float32')
    # probseg_img.header.set_slope_inter(1/(2**16-1),0.0)
    return probseg_img, dseg_img


def resample_atlas(fname,
                   atlas='MNISymC2',
                   target_space='MNI152NLin2009cSymC'):
    """ Resamples probabilistic atlas from MNISymC2 to a new atlas space 1mm resolution
    """
    a, ainf = am.get_atlas(atlas, ut.atlas_dir)
    src_dir = ut.model_dir + '/Atlases/'
    targ_dir = ut.base_dir + f'/Atlases/tpl-{target_space}'
    srcs_dir = ut.base_dir + '/Atlases/' + ainf['dir']
    nii_atlas = nb.load(src_dir + f'/{fname}_probseg.nii')
    # Reslice to 1mm MNI
    if ainf['space'] != target_space:
        print(f"deforming from {ainf['space']} to {target_space}")
        deform = nb.load(
            srcs_dir + f"/tpl-{ainf['space']}_space-{target_space}_xfm.nii")
        nii_res = nt.deform_image(nii_atlas, deform, 1)
    else:
        mname = ainf['mask']
        mname = mname.replace('res-2', 'res-1')
        nii_mask = nb.load(targ_dir + '/' + mname)
        # Make new shape
        shap = nii_mask.shape + nii_atlas.shape[3:]
        nii_res = ns.resample_from_to(nii_atlas, (shap, nii_mask.affine), 1)
    print('normalizing')
    nii, dnii = renormalize_probseg(nii_res)
    print('saving')
    nb.save(nii, targ_dir + f'/atl-{fname}_space-{target_space}_probseg.nii')
    nb.save(dnii, targ_dir + f'/atl-{fname}_space-{target_space}_dseg.nii')


def reorder_model(mname, sym=True, mname_new=None, assignment='mixed_assignment_68_16.csv', save_model=False):
    """
    Reorders a saved parcellation model according to fixed order assignment.

    Args:
        mname (str): The name of the saved model to be reordered.
        sym (bool): If True, reorders the model assuming symmetrical model.
        mname_new (str): The name of the reordered model. If None, the name will be the same as the original model with '_reordered' appended.
        assignment (str): The name of the CSV file containing the order assignment.
        save_model (bool): If True, saves the reordered model.

    Returns:
        new_model (object): The reordered model object.

    """
    # Get model and atlas.
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = ut.load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Get assignment
    assignment = pd.read_csv(
        f'{ut.model_dir}/Atlases/{assignment}')

    order_arrange = assignment['parcel_orig_idx'].values
    order_emission = np.concatenate(
        [order_arrange, order_arrange + len(order_arrange)])
    if not sym:
        order_arrange = order_emission

    # Reorder the model
    new_model = deepcopy(model)
    if new_model.arrange.logpi.shape[0] == order_arrange.shape[0]:
        new_model.arrange.logpi = model.arrange.logpi[order_arrange]
    elif new_model.arrange.logpi.shape[0] * 2 == order_arrange.shape[0]:
        new_model.arrange.logpi = model.arrange.logpi[order_arrange[:len(
            order_arrange) // 2]]
    else:
        raise ValueError(
            'The number of parcels in the model does not match the number of parcels in the assignment.')

    for e, em in enumerate(new_model.emissions):
        new_model.emissions[e].V = em.V[:, order_emission]

    # Info
    new_info = deepcopy(info)
    new_info['ordered_by'] = assignment
    new_info = new_info.to_frame().T

    # Save the model
    if save_model:
        if mname_new is None:
            mname_new = mname + '_reordered'
        # save new model
        with open(f'{ut.model_dir}/Models/{mname_new}.pickle', 'wb') as file:
            pickle.dump([new_model], file)

        # save new info
        new_info.to_csv(f'{ut.model_dir}/Models/{mname_new}.tsv',
                        sep='\t', index=False)

        print(
            f'Done. Saved reordered model as: \n\t{mname_new} \nOutput folder: \n\t{ut.model_dir}/Models/ \n\n')

    return new_model


def colour_parcel(mname, 
                  sym = False,
                  plot=True, 
                  labels=None, 
                  clusters=None, 
                  weighting=None, 
                  gamma=0):
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
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = ut.load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.arrange.marginal_prob())
    parcel = Prob.argmax(axis=0) + 1

    # Make a colormap.
    w_cos_sim, _, _ = cl.parcel_similarity(model,
                                           plot=True,
                                           sym=sym)
    W = sc.calc_mds(w_cos_sim, center=True)
    if sym:
        W = np.concatenate([W, W])
    # Define color anchors
    m, regions, colors = sc.get_target_points(atlas, parcel)
    cmap = sc.colormap_mds(W, target=(m, regions, colors),
                           clusters=clusters, gamma=gamma)
    sc.plot_colorspace(cmap(np.arange(model.K)))

    plt.figure(figsize=(5, 10))
    cl.plot_parcel_size(Prob, cmap, labels, wta=True)

    # Plot the parcellation
    if plot:
        ax = ut.plot_data_flat(Prob, atlas.name, cmap=cmap,
                               dtype='prob',
                               labels=labels,
                               render='plotly')
        ax.show()

    return Prob, parcel, atlas, labels, cmap
