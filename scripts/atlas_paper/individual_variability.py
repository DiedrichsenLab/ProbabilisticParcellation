"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
from Functional_Fusion.dataset import *
import ProbabilisticParcellation.util as ut
from copy import deepcopy
import ProbabilisticParcellation.learn_fusion_gpu as lf
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.functional_profiles as fp
import ProbabilisticParcellation.scripts.atlas_paper.symmetry as sm
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.evaluation as ev

pt.set_default_tensor_type(pt.FloatTensor)

# Correlate individual probabilistic parcellations pairwise
# Only for MDTB
# For 68 first, then for 32
# Normalize by inter-individual subject reliability

figure_path = "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/papers/AtlasPaper/figure_parts/"
if not os.path.exists(figure_path):
    figure_path = "/Users/callithrix/Dropbox/AtlasPaper/figure_parts/"
atlas_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/'


def correlate_parcellations(probs_indiv, dataset='MDTB', sym=None, norm=True):
    """Correlate individual parcellations 

    """
    # If dataset is given, use only dataset subjects
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    didx = np.where(T['name'] == dataset)[0][0]
    n_subj = T['return_nsubj'].iloc[didx]
    # Get subject indices for the dataset
    if dataset is not None:
        # Get dataset indices by adding the cumulative sum of subjects
        dataset_subjects = n_subj
        if didx > 1:
            dataset_subjects += np.sum(T['return_nsubj'].iloc[:didx])
        # Select only dataset subjects
        probs_indiv = probs_indiv[dataset_subjects -
                                  n_subj:dataset_subjects, :, :]

    # Get correlation for each voxel between subject's individual parcellations
    corr_mean = []
    for vox in np.arange(probs_indiv.shape[2]):
        corr = np.corrcoef(probs_indiv[:, :, vox])
        # Get upper triangular part indices
        corr = corr[np.triu_indices(corr.shape[0], k=1)]
        # Store mean
        corr_mean.append(np.nanmean(corr))

    # Make corr_mean list into numpy array
    corr_mean = np.array(corr_mean)

    if norm:
        filename = f'{sym}_indiv_var_{dataset}_{probs_indiv.shape[1]}_norm'
        reliability_voxelwise, _ = reliability_maps(ut.base_dir, dataset, atlas='MNISymC2',
                                                    subtract_mean=True, voxel_wise=True)
        # Get mean across sessions
        reliability_voxelwise = np.squeeze(
            np.nanmean(reliability_voxelwise, axis=0))
        # Set the negative reliability values to nan
        reliability_voxelwise[reliability_voxelwise < 0] = np.nan
        # Normalize correlation by reliability
        corr_mean = corr_mean / \
            np.sqrt(reliability_voxelwise)

        # corr_mean[~np.isnan(corr_mean)].min()
        # corr_mean[~np.isnan(corr_mean)].max()
        # print 95th percentile
        # print(np.percentile(corr_mean[~np.isnan(corr_mean)], 99))
    else:
        filename = f'{sym}_indiv_var_{dataset}_{probs_indiv.shape[1]}'

    # Plot corr_mean
    figsize = (8, 8)
    plt.figure(figsize=figsize)
    suit_atlas, _ = am.get_atlas('MNISymC2', ut.base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(corr_mean)
    surf_data = suit.flatmap.vol_to_surf(Nifti, stats='nanmean',
                                         space='MNISymC')

    suit.flatmap.plot(surf_data,
                      render='matplotlib',
                      new_figure=False,
                      cmap='hot',
                      cscale=(0, 1),
                      overlay_type='func',
                      colorbar=True,
                      bordersize=4)

    plt.savefig(
        f'{figure_path}/{filename}.png')
    pass


if __name__ == "__main__":
    norm = True
    sym = 'Asym'
    mname68 = f'Models_03/Nettekoven{sym}68_space-MNISymC2'
    # Get individual parcellations
    try:
        probs_indiv68 = pt.load(f'{ut.model_dir}/Models/{mname68}_Uhat.pt')
    except FileNotFoundError:
        probs_indiv68 = sm.export_uhats(
            mname=mname68)
    probs_indiv68 = probs_indiv68.numpy()

    mname32 = f'Models_03/Nettekoven{sym}32_space-MNISymC2'
    try:
        probs_indiv32 = pt.load(f'{ut.model_dir}/Models/{mname32}_Uhat.pt')
    except FileNotFoundError:
        probs_indiv32 = sm.export_uhats(
            mname=mname32)

    # Correlate individual probabilistic parcellations
    correlate_parcellations(probs_indiv68, dataset='MDTB', sym=sym, norm=norm)
    correlate_parcellations(probs_indiv32, dataset='MDTB', sym=sym, norm=norm)
    correlate_parcellations(
        probs_indiv32, dataset='Demand', sym=sym, norm=norm)
    correlate_parcellations(
        probs_indiv68, dataset='Demand', sym=sym, norm=norm)
    correlate_parcellations(
        probs_indiv32, dataset='Somatotopic', sym=sym, norm=norm)
    correlate_parcellations(
        probs_indiv68, dataset='Somatotopic', sym=sym, norm=norm)
    correlate_parcellations(probs_indiv32, dataset='WMFS', sym=sym, norm=norm)
    correlate_parcellations(probs_indiv68, dataset='WMFS', sym=sym, norm=norm)
    correlate_parcellations(probs_indiv32, dataset='IBC', sym=sym, norm=norm)
    correlate_parcellations(probs_indiv68, dataset='IBC', sym=sym, norm=norm)
