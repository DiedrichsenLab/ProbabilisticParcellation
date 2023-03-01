"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
from Functional_Fusion.dataset import *
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
import ProbabilisticParcellation.scripts.fit_C2_from_C3 as ft
import Functional_Fusion.dataset as ds
import generativeMRF.evaluation as ev
import logging

pt.set_default_tensor_type(pt.FloatTensor)


def correlate(X, Y):
    """ Correlate X and Y numpy arrays after standardizing them"""
    X = util.zstandarize_ts(X)
    Y = util.zstandarize_ts(Y)
    return Y.T @ X / X.shape[0]


def get_correlated_cortex(mname):
    """ Get the cortex correlated with the parcel profiles"""
    # Load model
    info, model = ut.load_batch_best(mname)
    info = fp.recover_info(info, model, mname)

    # Get parcel profile
    profile_file = f'{ut.model_dir}/Atlases/{mname.split("/")[-1]}_profile.tsv'
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
        Davg = np.nanmean(D, axis=0)
        if re.findall('[A-Z][^A-Z]*', info.type[d])[1] == 'Half':
            # TODO: Implement weighting of data according to kappa and sqrt(n_subjects)
            # Average across the two halves
            Davg = np.nanmean(
                np.stack([Davg[d_info.half == 1, :], Davg[d_info.half == 2, :]]), axis=0)
        dat.append(Davg)
    data = np.concatenate(dat, axis=0)

    # Correlate parcel profiles with cortical data
    cortex = correlate(data, profile)

    return cortex


def get_modelled_cortex(mname, mname_new=None, symmetry=None):
    model, info = ft.refit_model_in_new_space(
        mname, mname_new=mname_new, new_space='fs32k', symmetry=symmetry)
    cortex = model.arrange
    return cortex, info


def get_cortex(method='corr', mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed', symmetry=None):
    mname_new = f'{mname.split("/")[-1]}_cortex-{method}.dscalar.nii'

    # Get corresponding cortical parcels
    if method == 'corr':
        cortex = get_correlated_cortex(mname)
    elif method == 'model':
        cortex, info = get_modelled_cortex(
            mname, mname_new=mname_new, symmetry=symmetry)

    lut_file = ut.model_dir + '/Atlases/' + mname.split('/')[-1] + '.lut'
    if Path(lut_file).exists():
        index, cmap, labels = nt.read_lut(lut_file)

    # Get the fs32k atlas
    atlas, _ = am.get_atlas('fs32k', ut.atlas_dir)

    C = atlas.data_to_cifti(cortex, labels)
    nb.save(
        C, f'{ut.model_dir}/Atlases/{mname_new}')


if __name__ == "__main__":

    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'
    # cortex = get_cortex(mname=mname, method='corr')

    cortex = get_cortex(mname=mname, method='model', symmetry='sym')

    pass
