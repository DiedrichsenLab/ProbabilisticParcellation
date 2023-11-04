"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
import Functional_Fusion.dataset as ds
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
import os

pt.set_default_tensor_type(pt.FloatTensor)

figure_path = (
    "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/papers/AtlasPaper/figure_parts/"
)
if not os.path.exists(figure_path):
    figure_path = "/Users/callithrix/Dropbox/AtlasPaper/figure_parts/"
atlas_dir = (
    "/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/"
)


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
        # 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered',
        # 'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem'
        # "Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed"
        # "Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_arrange-asym_sep-hem_meth-mixed"
        # "Models_03/NettekovenSym68_space-MNISymC2",
        # "Models_03/NettekovenAsym68_space-MNISymC2",
        "Models_03/NettekovenSym32_space-MNISymC2",
        "Models_03/NettekovenAsym32_space-MNISymC2",
    ]
    # For reordering the parcels the first time (first merging)
    # f_assignment = "mixed_assignment_68_16.csv"
    # original_idx = "parcel_med_idx"

    # For reordering Introspection into action
    # f_assignment = "mixed_assignment_68_16_4.csv"
    # original_idx = "parcel_med_idx_5Domains"

    # For making action A4 into social S5
    f_assignment = "mixed_assignment_68_32_4.csv"

    for mname in mnames:

        if f_assignment=="mixed_assignment_68_16_4.csv":
            # For swapping M2 and A1
            if "32" in mname :
                original_idx = "parcel_med_before_tongue_swap_idx"
                mname_new = mname + "_tongueSwap"
            elif "68" in mname :
                # For swapping M2 and A1 in K68
                original_idx = "parcel_orig_idx_before_tongue_swap_idx"
                mname_new = mname + "_tongueSwap"
        elif f_assignment=="mixed_assignment_68_32_4.csv":
            if "32" in mname :
                original_idx = "parcel_med_before_a4_swap_idx"
                mname_new = mname + "_a4Swap"
            elif "68" in mname :
                # For swapping M2 and A1 in K68
                original_idx = "parcel_orig_idx_before_a4_swap_idx"
                mname_new = mname + "_a4Swap"
            
        
        symmetry = mname.split("/")[1].split("_")[0]
        if symmetry == "sym":
            sym = True
        else:
            sym = False
        model_reordered = ea.reorder_model(
            mname,
            sym=sym,
            assignment=f_assignment,
            original_idx=original_idx,
            save_model=True,
            mname_new = mname_new
        )

    pass



if __name__ == "__main__":
    # mname = 'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'
    # mname = 'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'
    # update_color_map(mname)

    # export_model_merged(
    #     mname_new='Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_arrange-asym_sep-hem_reordered_meth-mixed')

    # # --- Export merged model profile ---
    # mname = 'Models_03/NettekovenSym32_space-MNISymC2'
    # fileparts = mname.split('/')
    # split_mn = fileparts[-1].split('_')
    # info, model = ut.load_batch_best(mname)
    # index, cmap, labels = nt.read_lut(ut.base_dir + '/..//Cerebellum/ProbabilisticParcellationModel/Atlases/' +
    #                                   fileparts[-1].split('_')[0] + '.lut')
    # info = ut.recover_info(info, model, mname)
    # fp.export_profile(mname, info, model, labels)

    # # --- Export full 68 profile ---
    # mname = 'Models_03/NettekovenSym68_space-MNISymC2'
    # fileparts = mname.split('/')
    # split_mn = fileparts[-1].split('_')
    # info, model = ut.load_batch_best(mname)
    # index, cmap, labels = nt.read_lut(ut.base_dir + '/..//Cerebellum/ProbabilisticParcellationModel/Atlases/' +
    #                                   fileparts[-1].split('_')[0] + '.lut')
    # info = ut.recover_info(info, model, mname)
    # fp.export_profile(mname, info, model, labels)
    # features = fp.cognitive_features(mname)

    # # --- Reorder selected models according to our assignment ---
    reorder_selected()

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
