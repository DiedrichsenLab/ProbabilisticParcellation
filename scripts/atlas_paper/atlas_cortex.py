"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
from scipy.linalg import block_diag
import torch as pt
import matplotlib.pyplot as plt
import ProbabilisticParcellation.util as ut
import PcmPy as pcm
import torch as pt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from copy import deepcopy
import ProbabilisticParcellation.learn_fusion_gpu as lf
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.functional_profiles as fp
import ProbabilisticParcellation.scripts.atlas_paper.fit_C2_from_C3 as ft
import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am
import generativeMRF.evaluation as ev
import logging
import nitools as nt
import cortico_cereb_connectivity.scripts.script_plot_weights as cc
from pathlib import Path
import nibabel as nb

pt.set_default_tensor_type(pt.FloatTensor)


def correlate(X, Y):
    """ Correlate X and Y numpy arrays after standardizing them"""
    X = util.zstandarize_ts(X)
    Y = util.zstandarize_ts(Y)
    return Y.T @ X / X.shape[0]


def get_correlated_cortex(mname, weighting=False):
    """ Get the cortex correlated with the parcel profiles"""
    # Load model
    info, model = ut.load_batch_best(mname)
    info = fp.recover_info(info, model, mname)

    # Get parcel profile
    profile_file = f'{ut.model_dir}/Atlases/Profiles/{mname.split("/")[-1]}_profile.tsv'
    if Path(profile_file).exists():
        parcel_profiles = pd.read_csv(
            profile_file, sep="\t"
        )
    else:
        parcel_profiles, profile_data = fp.get_profiles(model, info)

    # Make profile into numpy array
    if isinstance(parcel_profiles, pd.DataFrame):
        idx_start = parcel_profiles.columns.tolist().index('condition') + 1
        parcel_names = parcel_profiles.columns[idx_start:idx_start + info.K]
        profile = parcel_profiles.iloc[:, idx_start:idx_start + info.K]
        profile = profile.values

    # Get cortical data
    dat = []
    for d, dataset in enumerate(info.datasets):

        D, d_info, dataset = ds.get_dataset(
            ut.base_dir, dataset, atlas='fs32k', sess=info.sess[d], type=info.type[d])
        D = np.nanmean(D, axis=0)
        if re.findall('[A-Z][^A-Z]*', info.type[d])[1] == 'Half':
            # Average across the two halves
            D = np.nanmean(
                np.stack([D[d_info.half == 1, :], D[d_info.half == 2, :]]), axis=0)

        # Weigh the V vectors by kappa (certainty) and the square root of the number of subjects
        # This is to make the profiles comparable across datasets with different number of subjects
        if weighting == True:
            em = model.emissions[d]
            D = D * em.kappa.item() * np.sqrt(em.num_subj)

        dat.append(D)
    data = np.concatenate(dat, axis=0)

    # Correlate parcel profiles with cortical data
    cortex = correlate(data, profile)

    return cortex


def get_modelled_cortex(mname, mname_new=None, symmetry=None):
    """ Get the cortex modelled from the parcel profiles"""
    if Path(
            f'{ut.model_dir}/Models/{mname.split("/")[0]}/{mname_new}.pickle').exists():
        info, model = ut.load_batch_best(
            f'{mname.split("/")[0]}/{mname_new}')
    else:
        model, info = ft.refit_model_in_new_space(
            mname, mname_new=mname_new, new_space='fs32k', symmetry=symmetry)
    cortex = model.arrange.marginal_prob()

    return cortex, info


def get_cortex(method='corr', mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed', symmetry=None):
    mname_new = f'{mname.split("/")[-1]}_cortex-{method}'
    if symmetry is not None:
        mname_new = f'{symmetry}_{mname_new.split("sym_")[1]}'

    # Get corresponding cortical parcels
    if method == 'corr':
        cortex = get_correlated_cortex(mname)
    elif method == 'model':
        cortex, info = get_modelled_cortex(
            mname, mname_new=mname_new, symmetry='asym')

    # lut_file = ut.model_dir + '/Atlases/' + mname.split('/')[-1] + '.lut'
    # if Path(lut_file).exists():
    #     index, cmap, labels = nt.read_lut(lut_file)
    # if labels[0] == '0':
    #     labels = labels[1:]

    # # Get the fs32k atlas
    # atlas, _ = am.get_atlas('fs32k', ut.atlas_dir)

    # C = atlas.data_to_cifti(cortex, labels)
    # nb.save(
    #     C, f'{ut.model_dir}/Atlases/{mname_new}.dscalar.nii')

    # prepping the parcel axis file
    atlas_fs, _ = am.get_atlas("fs32k", ut.atlas_dir)

    # load the label file for the cortex
    cortex_roi = 'Icosahedron1002'
    label_fs = [ut.atlas_dir +
                f"/tpl-fs32k/{cortex_roi}.{hemi}.label.gii" for hemi in ["L", "R"]]

    # get parcels for the neocortex
    _, label_fs = atlas_fs.get_parcel(label_fs, unite_struct=False)

    # get the average cortical weights for each cerebellar parcel
    atlas_suit, _ = am.get_atlas(mname.split(
        "space-")[1].split("_")[0], ut.atlas_dir)

    cortex_parcel, labels = ds.agg_parcels(
        cortex, atlas_fs.label_vector, fcn=np.nanmean)

    # preping the parcel axis
    # load the lookup table for the cerebellar parcellation to get the names of the parcels
    lut_file = ut.model_dir + '/Atlases/' + mname.split('/')[-1] + '.lut'
    assert Path(lut_file).exists(), 'No lut file found'
    index, cmap, parcel_labels = nt.read_lut(lut_file)
    # If labels don't end in L and R, add L to the first half and R to the second half
    if parcel_labels[0][-1] not in ['L', 'R']:
        parcel_labels = [l + 'L' if i <
                         len(parcel_labels) / 2 else l + 'R' for i, l in enumerate(parcel_labels)]

    # create parcel axis for the cortex (will be used as column axis in pscalar file)
    p_axis = atlas_fs.get_parcel_axis()

    # generate row axis with the last rowi being the scale
    row_axis = nb.cifti2.ScalarAxis(parcel_labels[1:])
    # Make torch.float32 data into numpy array
    data = cortex_parcel

    # make header
    # rows are maps corresponding to cerebellar parcels
    # columns are cortical tessels
    header = nb.Cifti2Header.from_axes((row_axis, p_axis))
    cifti_img = nb.Cifti2Image(data, header=header)
    cifti_img_new = cc.sort_roi_rows(cifti_img)
    nb.save(cifti_img_new, 'test_roi_sorted.pscalar.nii')
    # f'/{s}_space-fs32k_{ses_id}_{type}_Iso-{res}.pscalar.nii')
    return cifti_img


def export_cortex(mname):

    info, model = ut.load_batch_best(mname)
    info = fp.recover_info(info, model, mname)

    lut_file = ut.model_dir + '/Atlases/' + mname.split('/')[-1] + '.lut'
    if Path(lut_file).exists():
        index, cmap, labels = nt.read_lut(lut_file)

    base_name = f'{ut.model_dir}/Atlases/{mname_new}'

    if not isinstance(cmap, np.ndarray):
        cmap = cmap(np.arange(cmap.N))

    map_space = 'fs32k'
    atlas, _ = am.get_atlas(map_space, ut.atlas_dir)
    surf_probseg = model.arrange.marginal_prob()
    surf_parcel = np.argmax(surf_probseg, axis=0) + 1

    # get the gifti of the mask
    gii_mask = [nb.load(ut.atlas_dir + f'/tpl-fs32k/tpl-fs32k_hemi-L_mask.label.gii'),
                nb.load(ut.atlas_dir + f'/tpl-fs32k/tpl-fs32k_hemi-R_mask.label.gii')]

    gifti_img = []
    for i, name in zip([0, 1], ['CortexLeft', 'CortexRight']):
        # get data for the hemisphere
        data = surf_probseg[:, int(surf_probseg.shape[1] / 2)].T

        # get the labels for the hemisphere
        # half = int(len(labels[1:]) / 2)
        # labels_hem = labels[1:][:half]
        # labels_hem = [l[:2] for l in labels_hem]

        # # get the map for the contrast of interest
        # con_map = data[info_con.cond_num.values - 1, :]

        # # get threshold value (ignoring nans)
        # percentile_value = np.nanpercentile(con_map, q=threshold)

        # # apply threshold
        # thresh_data = con_map > percentile_value
        # # convert 0 to nan
        # thresh_data[thresh_data != False] = np.nan
        # create label gifti
        gifti_img.append(nt.make_label_gifti(
            data, anatomical_struct=name, label_names=labels[1:]))
    nb.save(gifti_img[0], filename='test.label.gii')

    Gifti = nt.make_label_gifti(surf_parcel.reshape(-1, 1),
                                anatomical_struct=[
                                    'CortexLeft', 'CortexRight'],
                                label_names=labels,
                                label_RGBA=cmap)
    # nb.save(dseg, base_name + f'_dseg.nii')
    # nb.save(probseg, base_name + f'_probseg.nii')
    nb.save(Gifti, base_name + '_dseg.label.gii')
    nt.save_lut(base_name, np.arange(len(labels)), cmap[:, 0:4], labels)
    print(f'Exported {base_name}.')

    for i, h in enumerate(['L', 'R']):
        nb.save(roi_gifti[i], save_dir + '/tpl-fs32k' +
                f'/vertical-{condition_1}_vs_{condition_2}.32k.{h}.label.gii')


if __name__ == "__main__":

    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'
    # mname = 'Models_03/NettekovenSym68c32'

    # -- Get correlated cortex --
    method = 'model'
    symmetry = 'asym'
    cortex = get_cortex(mname=mname, method=method, symmetry=symmetry)
    # # cortex = get_cortex(mname=mname, method=method, symmetry=symmetry)

    # mname_new = '_'.join(mname.split("/")[-1].split("_")[1:])
    # mname_new = f'Models_03/{symmetry}_{mname_new}_cortex-{method}'
    # export_cortex(mname=mname_new)

    pass
