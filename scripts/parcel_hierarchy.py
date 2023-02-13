"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
from Functional_Fusion.dataset import *
from scipy.linalg import block_diag
import torch as pt
import matplotlib.pyplot as plt
from ProbabilisticParcellation.util import *
import PcmPy as pcm
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from copy import deepcopy
import ProbabilisticParcellation.learn_fusion_gpu as lf
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.functional_profiles as fp
import Functional_Fusion.dataset as ds
import generativeMRF.evaluation as ev
import logging

pt.set_default_tensor_type(pt.FloatTensor)


def analyze_parcel(mname, sym=True, num_cluster=5, clustering='agglomative', cluster_by=None, plot=True, labels=None):

    # Get model and atlas.
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.arrange.marginal_prob())
    parcel = Prob.argmax(axis=0) + 1

    # Get parcel similarity:
    w_cos_sym, _, _ = cl.parcel_similarity(model, plot=True, sym=sym)

    # Do Clustering:
    if clustering == 'agglomative':
        labels_new, clusters, leaves = cl.agglomative_clustering(
            w_cos_sym, sym=sym, num_clusters=num_cluster, plot=False)
        while np.unique(clusters).shape[0] < 2:
            num_cluster = num_cluster - 1
            labels_new, clusters, _ = cl.agglomative_clustering(
                w_cos_sym, sym=sym, num_clusters=num_cluster, plot=False)
            logging.warning(
                f' Number of desired clusters too small to find at least two clusters. Set number of clusters to {num_cluster} and found {np.unique(clusters).shape[0]} clusters')
    if clustering == 'model_guided':
        if cluster_by is None:
            raise ('Need to specify model that guides clustering')
        cluster_info, cluster_model = load_batch_best(cluster_by)
        clusters_half, clusters = cl.guided_clustering(
            mname, cluster_by)
        labels, cluster_counts = cl.cluster_labels(clusters)
        print(
            f'Found {len(cluster_counts)} clusters with no of regions: {cluster_counts}\n')

    # Make a colormap...
    w_cos_sim, _, _ = cl.parcel_similarity(model, plot=False)
    W = sc.calc_mds(w_cos_sim, center=True)

    # Define color anchors
    m, regions, colors = sc.get_target_points(atlas, parcel)
    cmap = sc.colormap_mds(W, target=(m, regions, colors),
                           clusters=clusters, gamma=0)
    sc.plot_colorspace(cmap(np.arange(model.K)))

    # Replot the Clustering dendrogram, this time with the correct color map
    if clustering == 'agglomative':
        if labels is None:
            labels = labels_new
            cl.agglomative_clustering(
                w_cos_sym, sym=sym, num_clusters=num_cluster, plot=True, cmap=cmap)

    plt.figure(figsize=(5, 10))
    cl.plot_parcel_size(Prob, cmap, labels, wta=True)

    # Plot the parcellation
    if plot:
        ax = plot_data_flat(Prob, atlas.name, cmap=cmap,
                            dtype='prob',
                            labels=labels,
                            render='plotly')
        ax.show()

    return Prob, parcel, atlas, labels, cmap


def merge_clusters(ks, space='MNISymC3'):
    # save_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/Models/'
    # --- Merge parcels at K=20, 34 & 40 ---
    merged_models = []

    for k in ks:

        mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'
        mname_coarse = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-{k}'

        # merge model
        _, mname_merged = cl.cluster_parcel(
            mname_fine, mname_coarse)
        merged_models.append(mname_merged)
    return merged_models


def export_merged(merged_models=None):

    # --- Export merged models ---
    if merged_models is None:
        merged_models = [
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-10_Keff-10',
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-14_Keff-14',
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-20_Keff-20',
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-28_Keff-22',
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-34_Keff-24',
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-40_Keff-24',
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-48_Keff-36',
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-56_Keff-36',
            f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-60_Keff-38']

    for mname_merged in merged_models:
        # export the merged model
        Prob, parcel, atlas, labels, cmap = analyze_parcel(
            mname_merged, sym=True)
        ea.export_map(Prob, atlas.name, cmap, labels,
                      f'{model_dir}/Atlases/{mname_merged.split("/")[1]}')


def compare_levels():
    """Compares different clustering levels.
        For a selection of merged models, calculate adjusted Rand index between original fine parcellation and merged parcellation (merged according to coarse parcellation).

    """
    # Compare original parcellation with clustered parcellation
    atlas = 'MNISymC2'

    fine_model = f'/Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68'
    fileparts = fine_model.split('/')
    split_mn = fileparts[-1].split('_')
    info_68, model_68 = load_batch_best(fine_model)
    Prop_68 = np.array(model_68.marginal_prob())
    parcel_68 = Prop_68.argmax(axis=0) + 1

    merged_models = [
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-10_Keff-10',
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-14_Keff-14',
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-20_Keff-20',
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-28_Keff-22',
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-34_Keff-24',
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-40_Keff-24',
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-48_Keff-36',
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-56_Keff-36',
        f'Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68_Kclus-60_Keff-38']

    m_models = []
    m_infos = []
    for mname in merged_models:
        info, model = load_batch_best(mname)
        m_models.append(model)
        m_infos.append(info)

        n_models = len(m_models)
        n_voxels = parcel_68.shape[0]
        m_parcels = np.zeros((n_models, n_voxels))

        for i, model in enumerate(m_models):
            Prop = np.array(model.marginal_prob())
            parcel = Prop.argmax(axis=0) + 1
            m_parcels[i, :] = parcel

        # get U_hat

        ev.ARI(parcel_68, m_parcels[0, :])
        parcel
        pass


def save_pmaps(Prob, labels, atlas, subset=[0, 1, 2, 3, 4, 5]):
    plt.figure(figsize=(7, 10))
    plot_model_pmaps(Prob, atlas,
                     labels=labels[1:],
                     subset=subset,
                     grid=(3, 2))
    plt.savefig(f'pmaps_01.png', format='png')
    pass


def query_similarity(mname, label):
    labels, w_cos_sim, spatial_sim, ind = similarity_matrices(mname)
    ind = np.nonzero(labels == label)[0]
    D = pd.DataFrame(
        {'labels': labels, 'w_cos_sim': w_cos_sim[ind[0], :], 'spatial_sim': spatial_sim[ind[0], :]})
    return D


def plot_model_taskmaps(mname, n_highest=3, n_lowest=2, datasets=['Somatotopic', 'Demand'], save_task_maps=False):
    """Plots taskmaps of highest and lowest scoring tasks for each parcel
    Args:
        n_tasks: Number of tasks to save

    Returns:
        fig: task map plot

    """
    profile = pd.read_csv(
        f'{model_dir}/Atlases/{mname.split("/")[-1]}_task_profile_data.tsv', sep="\t"
    )
    atlas = mname.split('space-')[1].split('_')[0]
    Prob, parcel, _, labels, cmap = analyze_parcel(mname, sym=True)
    labels_sorted = sorted(labels)

    for dataset in datasets:
        # Select dataset
        dataset = datasets[0]
        profile_dataset = profile[profile.dataset == dataset]

        # Get highest scoring task of this dataset
        conditions = profile_dataset.condition
        tasks = {}
        for region in labels_sorted[1:]:
            weights = profile_dataset[region]
            conditions_weighted = [(con, w)
                                   for con, w in sorted(zip(weights, conditions), reverse=True)]
            high = conditions_weighted[:n_highest]
            low = conditions_weighted[-n_lowest:]
            tasks[region] = high + low
            print(
                f'\n\n\n{region}\nHighest: \t{[el[1] for el in high]}\n\t\t{[el[0] for el in high]}\n\nLowest: \t{[el[1] for el in low]}\n\t\t{[el[0] for el in low]}')

        if save_task_maps:
            # Get task maps
            data, info, _ = ds.get_dataset(base_dir, dataset, atlas=atlas)
            grid = (int(np.ceil((n_highest + n_lowest) / 2)), 2)
            for region in tasks.keys():
                task = tasks[region]
                activity = np.full((len(task), data.shape[2]), np.nan)
                for i, t in enumerate(task):
                    task_name = t[1]
                    activity[i, :] = np.nanmean(
                        data[:, info.cond_name.tolist().index(task_name), :], axis=0)

                titles = [
                    f'{name} V: {np.round(weight,2)}' for weight, name in task]

                cscale = [np.percentile(activity[np.where(~np.isnan(activity))], 5), np.percentile(
                    activity[np.where(~np.isnan(activity))], 95)]
                plot_multi_flat(activity, atlas, grid,
                                cmap='hot',
                                dtype='func',
                                cscale=cscale,
                                titles=titles,
                                colorbar=False,
                                save_fig=True,
                                save_under=f'task_maps/{dataset}_{region}.png')

    pass


def save_taskmaps(mname):
    """Saves taskmaps of highest and lowest scoring tasks for each parcel
    Args:
        n_tasks: Number of tasks to save
        datasets:
    """
    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'

    plt.figure(figsize=(7, 10))
    plot_model_taskmaps(mname, n_highest=3, n_lowest=3)
    plt.savefig(f'tmaps_01.png', format='png')


def make_NettekovenSym68c32():
    space = 'MNISymC2'
    mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'
    mname_new = 'Models_03/NettekovenSym68c32'
    f_assignment = 'mixed_assignment_68_16.csv'
    _, _, labels = save_mixed_clustering(mname_fine, method='mixed',
                                         mname_new=mname_new,
                                         f_assignment='mixed_assignment_68_16.csv',
                                         refit_model=True, save_model=True)

    Prob, parcel, atlas, labels, cmap = analyze_parcel(
        mname_new, sym=True, labels=labels)
    ea.export_map(Prob, atlas.name, cmap, labels,
                  f'{model_dir}/Atlases/{mname_new.split("/")[1]}')
    ea.resample_atlas('NettekovenSym68c32',
                      atlas='MNISymC2',
                      target_space='SUIT')


def profile_NettekovenSym68c32():
    space = 'MNISymC2'
    mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'
    mname_new = 'Models_03/NettekovenSym68c32'
    f_assignment = 'mixed_assignment_68_16.csv'
    _, _, labels = save_mixed_clustering(mname_fine, method='mixed',
                                         mname_new=mname_new,
                                         f_assignment='mixed_assignment_68_16.csv',
                                         refit_model=False, save_model=False)

    Prob, parcel, atlas, labels, cmap = analyze_parcel(
        mname_new, sym=True, labels=labels)
    # save_pmaps(Prob, labels, space, subset=[0, 1, 2, 3, 4, 5])
    # save_pmaps(Prob, labels, space, subset=[6, 7, 8, 9, 10, 11])
    # save_pmaps(Prob, labels, space, subset=[12, 13, 14, 15])
    info, model = load_batch_best(mname_new)
    info = fp.recover_info(info, model, mname_new)
    fp.export_profile(mname_new, info, model, labels)
    features = fp.cognitive_features(mname_new)


if __name__ == "__main__":
    # make_NettekovenSym68c32()
    profile_NettekovenSym68c32()
    # ea.resample_atlas('NettekovenSym68c32',
    #                   atlas='MNISymC2',
    #                   target_space='MNI152NLin6AsymC')
    # Save 3 highest and 2 lowest task maps
    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    # D = query_similarity(mname, 'E3L')
    # save_taskmaps(mname)

    # Merge functionally and spatially clustered scree parcels

    # index, cmap, labels = nt.read_lut(model_dir + '/Atlases/' +
    #                                   fileparts[-1] + '.lut')
    # get data

    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    # f_assignment = 'mixed_assignment_68_16.csv'
    # df_assignment = pd.read_csv(
    #     model_dir + '/Atlases/' + '/' + f_assignment)

    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    # f_assignment = 'mixed_assignment_68_16.csv'
    # df_assignment = pd.read_csv(
    #     model_dir + '/Atlases/' + '/' + f_assignment)

    # mapping, labels = mixed_clustering(mname, df_assignment)

    pass
