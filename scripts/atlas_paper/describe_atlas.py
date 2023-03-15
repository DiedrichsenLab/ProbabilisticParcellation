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
import ProbabilisticParcellation.scripts.atlas_paper.fit_C2_from_C3 as ft
import Functional_Fusion.dataset as ds
import generativeMRF.evaluation as ev
# from ProbabilisticParcellation.scripts.atlas_paper.parcel_hierarchy import analyze_parcel
import logging
import nitools as nt

pt.set_default_tensor_type(pt.FloatTensor)


def export_asym_from_sym():
    space = 'MNISymC2'
    mname = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68_arrange-asym'
    f_assignment = 'mixed_assignment_68_16.csv'
    assignment = pd.read_csv(
        f'{ut.model_dir}/Atlases/{f_assignment}')

    labels = assignment['parcel_fine']
    labels_left = labels + 'L'
    labels_right = labels + 'R'
    labels = ['0'] + labels_left.tolist() + labels_right.tolist()

    order = assignment['parcel_orig_idx'].values
    order = np.concatenate([order, order + 34])
    cluster_names, clusters = np.unique(
        assignment['domain'], return_inverse=True)
    clusters = clusters + 1
    clusters = np.concatenate([clusters, clusters])

    Prob, parcel, atlas, labels, cmap = ea.colour_parcel(
        mname, sym=False, labels=labels, clusters=clusters, reorder=order)

    # Get colour map from symmetric 68 model
    _, cmap_sym, _ = nt.read_lut(ut.model_dir + '/Atlases/' +
                                 f'sym_MdPoNiIbWmDeSo_space-{space}_K-68' + '.lut')
    cmap_sym = ListedColormap(cmap_sym)

    ea.export_map(Prob, atlas.name, cmap_sym, labels,
                  f'{ut.model_dir}/Atlases/{mname.split("/")[1]}')
    print('Exported atlas to ' +
          f'{ut.model_dir}/Atlases/{mname.split("/")[1]}')
    pass


def export_orig_68():
    space = 'MNISymC2'
    mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'

    Prob, parcel, atlas, labels, cmap = ea.analyze_parcel(
        mname_fine, sym=True)
    ea.export_map(Prob, atlas.name, cmap, labels,
                  f'{ut.model_dir}/Atlases/{mname_fine.split("/")[1]}')


def make_NettekovenSym68c32():
    space = 'MNISymC2'
    mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'
    mname_new = 'Models_03/NettekovenSym68c32'
    f_assignment = 'mixed_assignment_68_16.csv'
    _, _, labels = cl.cluster_parcel(mname_fine, method='mixed',
                                     mname_new=mname_new,
                                     f_assignment='mixed_assignment_68_16.csv',
                                     refit_model=True, save_model=True)

    Prob, parcel, atlas, labels, cmap = ea.analyze_parcel(
        mname_new, sym=True, labels=labels)
    ea.export_map(Prob, atlas.name, cmap, labels,
                  f'{ut.model_dir}/Atlases/{mname_new.split("/")[1]}')
    ea.resample_atlas('NettekovenSym68c32',
                      atlas='MNISymC2',
                      target_space='SUIT')


def profile_NettekovenSym68c32():
    space = 'MNISymC2'
    mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'
    mname_new = 'Models_03/NettekovenSym68c32'
    f_assignment = 'mixed_assignment_68_16'
    _, _, labels = cl.cluster_parcel(mname_fine, method='mixed',
                                     mname_new=mname_new,
                                     f_assignment=f_assignment,
                                     refit_model=False, save_model=False)

    Prob, parcel, atlas, labels, cmap = ea.analyze_parcel(
        mname_new, sym=True, labels=labels)
    save_pmaps(Prob, labels, space, subset=[0, 1, 2, 3, 4, 5])
    save_pmaps(Prob, labels, space, subset=[6, 7, 8, 9, 10, 11])
    save_pmaps(Prob, labels, space, subset=[12, 13, 14, 15])
    info, model = ut.load_batch_best(mname_new)
    info = fp.recover_info(info, model, mname_new)
    fp.export_profile(mname_new, info, model, labels)
    features = fp.cognitive_features(mname_new)


def export_model_merged(mname_new):
    space = mname_new.split('space-')[1].split('_')[0]
    mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'
    f_assignment = 'mixed_assignment_68_16'
    _, _, labels = cl.cluster_parcel(mname_fine, method='mixed',
                                     mname_new=mname_new,
                                     f_assignment=f_assignment,
                                     refit_model=False, save_model=False)

    Prob, parcel, atlas, labels, cmap = ea.analyze_parcel(
        mname_new, sym=True, labels=labels)
    save_pmaps(Prob, labels, space, subset=[0, 1, 2, 3, 4, 5])
    save_pmaps(Prob, labels, space, subset=[6, 7, 8, 9, 10, 11])
    save_pmaps(Prob, labels, space, subset=[12, 13, 14, 15])
    info, model = ut.load_batch_best(mname_new)
    info = fp.recover_info(info, model, mname_new)

    ea.export_map(Prob, atlas.name, cmap, labels,
                  f'{ut.model_dir}/Atlases/{mname_new.split("/")[1]}')


def save_pmaps(Prob, labels, atlas, subset=[0, 1, 2, 3, 4, 5]):
    plt.figure(figsize=(7, 10))
    ut.plot_model_pmaps(Prob, atlas,
                        labels=labels[1:],
                        subset=subset,
                        grid=(3, 2))
    plt.savefig(f'pmaps_01.png', format='png')
    pass


if __name__ == "__main__":

    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'

    # # --- Export merged model profile ---
    # fileparts = mname.split('/')
    # split_mn = fileparts[-1].split('_')
    # info, model = ut.load_batch_best(mname)
    # index, cmap, labels = nt.read_lut(ut.model_dir + '/Atlases/' +
    #                                   fileparts[-1] + '.lut')
    # info = fp.recover_info(info, model, mname)
    # fp.export_profile(mname, info, model, labels)

    # features = fp.cognitive_features(mname)

    # --- Export asymmetric model fitted from symmetric model ---
    export_asym_from_sym()

    pass
