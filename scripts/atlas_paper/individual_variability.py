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
import ProbabilisticParcellation.scripts.atlas_paper.describe_atlas as da
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nilearn.plotting as nip


def subset_probs(probs_indiv, dataset):
    """Subset individual parcellations to a specific dataset"""

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    didx = np.where(T['name'] == dataset)[0][0]
    n_subj = T['return_nsubj'].iloc[didx]

    dataset_subjects = n_subj
    if didx > 1:
        dataset_subjects += np.sum(T['return_nsubj'].iloc[:didx])

    return probs_indiv[dataset_subjects - n_subj:dataset_subjects, :, :]


def inter_individual_variability(probs_indiv):
    """Get inter-individual variability between individual parcellations by correlating probablistic parcellations voxelwise"""
    corr_mean = []
    for vox in np.arange(probs_indiv.shape[2]):
        corr = np.corrcoef(probs_indiv[:, :, vox])
        corr = corr[np.triu_indices(corr.shape[0], k=1)]
        corr_mean.append(np.nanmean(corr))

    return np.array(corr_mean)


def reliability_norm(corr_mean, dataset, probs_indiv):
    """Normalize inter-individual variability by intra-individual reliability"""
    reliability_voxelwise, _ = reliability_maps(ut.base_dir, dataset, atlas='MNISymC2',
                                                subtract_mean=True, voxel_wise=True)
    reliability_voxelwise = np.squeeze(
        np.nanmean(reliability_voxelwise, axis=0))
    reliability_voxelwise[reliability_voxelwise < 0] = np.nan
    corr_mean = corr_mean / np.sqrt(reliability_voxelwise)
    return corr_mean


def plot_variability(corr_mean, filename, save=True):
    """Plot inter-individual variability and save to file"""
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
    if save:
        plt.savefig(f'{figure_path}/individual_variability/{filename}.png')


def describe_variability():

    norm = True
    sym = 'Asym'
    K = 32
    space = 'MNISymC2'

    mname = f'Models_03/Nettekoven{sym}{K}_space-{space}'

    # Get individual parcellations
    try:
        probs_indiv = pt.load(f'{ut.model_dir}/Models/{mname}_Uhat.pt')
        probs_info = pd.read_csv(
            f'{ut.model_dir}/Models/{mname}_Uhat_info.tsv', sep='\t')
    except FileNotFoundError:
        probs_indiv, probs_info = sm.export_uhats(
            mname=mname)
    probs_indiv = probs_indiv.numpy()

    # Describe inter-individual variability for all datasets
    datasets = probs_info['dataset'].unique()
    Corr = []
    for dataset in datasets:
        probs_indiv = subset_probs(probs_indiv, dataset)
        probs_indiv[probs_info[probs_info == probs_info], :, :]

        corr_mean = inter_individual_variability(probs_indiv)
        Corr.append(corr_mean)

        if norm:
            corr_mean = reliability_norm(corr_mean, dataset, probs_indiv)
            filename = f'{sym}_indiv_var_{dataset}_{probs_indiv.shape[1]}_norm'
        else:
            filename = f'{sym}_indiv_var_{dataset}_{probs_indiv.shape[1]}'

        plot_variability(corr_mean, filename, save=True)

    # Plot all variabilities as grid
    Corr = np.array(Corr)
    plt.figure(figsize=(14, 8))
    ut.plot_multi_flat(Corr, space,
                       grid=(1, 2),
                       dtype='func',
                       colorbar=True,
                       titles=datasets)


def plot_dataset_pmaps(plot_parcels=['M1', 'M3', 'D1', 'D2', 'D3', 'D4']):
    """Get individual Uhats, average across the Uhats and plot as pmaps"""
    K = 32
    space = 'MNISymC2'
    mname = f'Models_03/NettekovenSym{K}_space-{space}'

    # Get individual parcellations
    try:
        probs_indiv = pt.load(f'{ut.model_dir}/Models/{mname}_Uhat.pt')
    except FileNotFoundError:
        probs_indiv = sm.export_uhats(
            mname=mname)
    probs_indiv = probs_indiv.numpy()

    fileparts = mname.split('/')
    index, cmap, labels = nt.read_lut(ut.base_dir + '/..//Cerebellum/ProbabilisticParcellationModel/Atlases/' +
                                      fileparts[-1].split('_')[0] + '.lut')

    labels = labels[1:]
    # Make pmaps for each dataset
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    datasets = T['name'].unique()
    for dataset in datasets:
        probs_dataset = subset_probs(probs_indiv, dataset)
        # Mean prob across subjects for this dataset
        Prob = np.mean(probs_dataset, axis=0)
        subset = [labels.index(p + 'L') for p in plot_parcels]
        parc_names = "-".join([str(p) for p in plot_parcels])
        fig = da.save_pmaps(Prob, labels,
                            space, subset=subset, filename=f'{figure_path}/pmaps_{dataset}_{parc_names}.png')

    pass


if __name__ == "__main__":

    describe_variability()
    # plot_dataset_pmaps(['M1', 'M3', 'D1', 'D2', 'D3', 'D4'])
