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
import HierarchBayesParcel.evaluation as ev
import ProbabilisticParcellation.evaluate as ppev
# from ProbabilisticParcellation.scripts.atlas_paper.parcel_hierarchy import analyze_parcel
import logging
import nitools as nt

pt.set_default_tensor_type(pt.FloatTensor)


def reorder_selected():
    mnames = [
        # 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68',
        # 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68',
        # 'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem'
        # 'Models_03/sym_Md_space-MNISymC3_K-68',
        # 'Models_03/sym_Po_space-MNISymC3_K-68',
        # 'Models_03/sym_Ni_space-MNISymC3_K-68',
        # 'Models_03/sym_Ib_space-MNISymC3_K-68',
        # 'Models_03/sym_Wm_space-MNISymC3_K-68',
        # 'Models_03/sym_De_space-MNISymC3_K-68',
        # 'Models_03/sym_So_space-MNISymC3_K-68',
        # 'Models_03/sym_Hc_space-MNISymC3_K-68',
        'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'
    ]

    f_assignment = 'mixed_assignment_68_16.csv'
    for mname in mnames:
        symmetry = mname.split('/')[1].split('_')[0]
        if symmetry == 'sym':
            sym = True
        else:
            sym = False
        model_reordered = ea.reorder_model(
            mname, sym=sym, assignment=f_assignment, save_model=True)


def export_selected():
    mnames = [
        # 'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC3_K-68',
        # 'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68',
        # 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered',
        'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem'
    ]
    for mname in mnames:
        export(mname)


def export(mname, sym=False):

    space = mname.split('space-')[1].split('_')[0]
    symmetry = mname.split('/')[1].split('_')[0]
    if symmetry == 'sym':
        sym = True
    f_assignment = 'mixed_assignment_68_16.csv'

    # Get assigned labels & clusters
    assignment = pd.read_csv(
        f'{ut.model_dir}/Atlases/{f_assignment}')

    labels = assignment['parcel_fine']
    cluster_names, clusters = np.unique(
        assignment['domain'], return_inverse=True)
    clusters = clusters + 1
    clusters = np.concatenate([clusters, clusters])

    # Extend symmetric labels to both hemispheres
    labels_left = labels + 'L'
    labels_right = labels + 'R'
    labels = ['0'] + labels_left.tolist() + labels_right.tolist()

    Prob, parcel, atlas, labels, cmap = ea.colour_parcel(
        mname, labels=labels, clusters=clusters)

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


def update_color_map(mname,):
    atlas_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/'
    _, cmap, labels = nt.read_lut(atlas_dir + mname + '.lut')
    Prob, parcel, atlas, labels, cmap = ea.colour_parcel(
        mname='Models_03/' + mname,
        sym=True,
        labels=labels)
    cmap_array = np.array(cmap(np.arange(len(labels))))
    nt.save_lut(atlas_dir + '/' + mname + '.lut',
                np.arange(len(labels)), cmap_array[:, 0:4], labels)


def export_model_merged(mname_new):
    space = mname_new.split('space-')[1].split('_')[0]
    # mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'
    mname_fine = f'Models_03/asym_MdPoNiIbWmDeSo_space-{space}_K-68_arrange-asym_sep-hem'
    f_assignment = 'mixed_assignment_68_16'
    # _, _, labels = cl.cluster_parcel(mname_fine, method='mixed',
    #                                  mname_new=mname_new,
    #                                  f_assignment=f_assignment,
    #                                  refit_model=True, save_model=True)

    mname_clus = f'Models_03/asym_MdPoNiIbWmDeSo_space-{space}_K-32_arrange-asym_sep-hem_meth-mixed'
    info, model = ut.load_batch_best(mname_clus)
    Prob = model.marginal_prob().numpy()
    atlas, _ = am.get_atlas(space, ut.atlas_dir)
    _, cmap, labels = nt.read_lut(ut.model_dir + '/Atlases/' +
                                  'NettekovenSym32.lut')
    Prob, parcel, atlas, labels, cmap = ea.colour_parcel(
        mname=mname_clus,
        sym=False,
        labels=labels)
    cmap_array = np.array(cmap(np.arange(len(labels))))

    # Prob, parcel, atlas, labels, cmap = ea.analyze_parcel(
    #     mname_new, sym=True, labels=labels)
    # save_pmaps(Prob, labels, space, subset=[0, 1, 2, 3, 4, 5])
    # save_pmaps(Prob, labels, space, subset=[6, 7, 8, 9, 10, 11])
    # save_pmaps(Prob, labels, space, subset=[12, 13, 14, 15])

    # save_pmaps(Prob, labels, space, subset=[16, 17, 18, 19, 20])
    # save_pmaps(Prob, labels, space, subset=[21, 22, 23, 24, 25 ])
    # save_pmaps(Prob, labels, space, subset=[26, 27, 28, 29, 30, 31])
    # save_pmaps(Prob, labels, space, subset=[32, 33])

    # info, model = ut.load_batch_best(mname_new)
    # info = fp.recover_info(info, model, mname_new)

    ea.export_map(Prob, atlas.name, cmap_array[:, 0:4], labels,
                  f'{ut.model_dir}/Atlases/{mname_new.split("/")[1]}')


def save_pmaps(Prob, labels, atlas, subset=[0, 1, 2, 3, 4, 5]):
    plt.figure(figsize=(7, 10))
    ut.plot_model_pmaps(Prob, atlas,
                        labels=labels[1:],
                        subset=subset,
                        grid=(3, 2))
    subset_name = "-".join([str(s) for s in subset])
    plt.savefig(f'pmaps_{subset_name}.png', format='png')
    pass


if __name__ == "__main__":

    # mname = 'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'
    # mname = 'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'
    # update_color_map(mname)

    # export_model_merged(
    #     mname_new='Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_arrange-asym_sep-hem_reordered_meth-mixed')

    # # --- Export merged model profile ---
    mname = 'Models_03/NettekovenSym32_space-MNISymC2'
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = ut.load_batch_best(mname)
    index, cmap, labels = nt.read_lut(ut.base_dir + '/..//Cerebellum/ProbabilisticParcellationModel/Atlases/' +
                                      fileparts[-1].split('_')[0] + '.lut')
    info = ut.recover_info(info, model, mname)
    fp.export_profile(mname, info, model, labels)

    # # --- Export full 68 profile ---
    mname = 'Models_03/NettekovenSym68_space-MNISymC2'
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = ut.load_batch_best(mname)
    index, cmap, labels = nt.read_lut(ut.base_dir + '/..//Cerebellum/ProbabilisticParcellationModel/Atlases/' +
                                      fileparts[-1].split('_')[0] + '.lut')
    info = ut.recover_info(info, model, mname)
    fp.export_profile(mname, info, model, labels)
    # features = fp.cognitive_features(mname)

    # # --- Reorder selected models according to our assignment ---
    # reorder_selected()
    # # --- Export asymmetric model fitted from symmetric model ---
    # export_selected()
    # # --- Export individual parcellations ---
    # export_uhats()
    # --- Export ARIs ---
    # load Uhats
    # model_pair = ['Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered',
    #               'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem_reordered']

    # prob_a = pt.load(f'{ut.model_dir}/Models/{model_pair[0]}_Uhat.pt')
    # prob_b = pt.load(f'{ut.model_dir}/Models/{model_pair[1]}_Uhat.pt')
    # parcel_a = pt.argmax(prob_a, dim=1)
    # parcel_b = pt.argmax(prob_b, dim=1)

    # # Load model
    # info_a, model_a = ut.load_batch_best(model_pair[0])

    # ------ Calculate comparison ------
    # comparison, comparison_group = ppev.ARI_voxelwise(
    #     parcel_a, parcel_b).numpy()
    # comparison, comparison_group = ppev.ARI_voxelwise(
    #     parcel_a, parcel_b, adjusted=False).numpy()
    # comparison, comparison_group = ppev.compare_probs(
    #     prob_a, prob_b, method='corr')

    # ax = ut.plot_multi_flat([comparison_group], 'MNISymC2',
    #                         grid=(1, 1),
    #                         dtype='func',
    #                         cmap='RdYlBu_r',
    #                         cscale=[0, 1],
    #                         colorbar=True)
    # comparison = ppev.compare_probs(
    #     prob_a, prob_b, method='cosang')

    # export_uhats(mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68')
    # export_uhats(
    #     mname='Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem')
