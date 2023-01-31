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
import generativeMRF.evaluation as ev
import logging

pt.set_default_tensor_type(pt.FloatTensor)


def analyze_parcel(mname, sym=True, num_cluster=5, clustering='agglomative', cluster_by=None, plot=True):

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
        labels, clusters, leaves = cl.agglomative_clustering(
            w_cos_sym, sym=sym, num_clusters=num_cluster, plot=False)
        while np.unique(clusters).shape[0] < 2:
            num_cluster = num_cluster - 1
            labels, clusters, _ = cl.agglomative_clustering(
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

    # Make a colormap
    w_cos_sim, _, _ = cl.parcel_similarity(model, plot=False)
    W = sc.calc_mds(w_cos_sim, center=True)

    # Define color anchors
    m, regions, colors = sc.get_target_points(atlas, parcel)
    cmap = sc.colormap_mds(W, target=(m, regions, colors),
                           clusters=clusters, gamma=0)
    sc.plot_colorspace(cmap(np.arange(model.K)))

    # Replot the Clustering dendrogram, this time with the correct color map
    if clustering == 'agglomative':
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
        _, mname_merged = cl.save_guided_clustering(
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


def save_pmaps(mname):
    Prob, parcel, atlas, labels, cmap = analyze_parcel(mname, sym=True)
    plt.figure(figsize=(7, 10))
    plot_model_pmaps(Prob, atlas.name,
                     labels=labels[1:],
                     subset=[30, 31, 32, 33, 33, 33],
                     grid=(3, 2))
    plt.savefig(f'pmaps_01.png', format='png')


def similarity_matrices(mname, sym=True):
    # Get model and atlas.
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.arrange.marginal_prob())
    index, cmap, labels = nt.read_lut(model_dir + '/Atlases/' +
                                      fileparts[-1] + '.lut')
    K, P = Prob.shape
    if sym:
        K = int(K / 2)
        Prob = Prob[:K, :]
    labels = labels[1:K + 1]

    # Get parcel similarity:
    w_cos_sym, _, _ = cl.parcel_similarity(model, plot=False, sym=sym)
    P = Prob / np.sqrt(np.sum(Prob**2, axis=1).reshape(-1, 1))

    spatial_sym
    l = labels[1:K]
    return


if __name__ == "__main__":
    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    similarity_matrices(mname)
    #
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # save_pmaps(mname)

    # Merge C2 models
    # space='MNISymC2'
    # mname_fine = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68'
    # mname_coarse = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-40'
    # cl.guided_clustering(mname_fine, mname_coarse,'cosang')
    # pass

    # export_merged(merged_models)

    # export_merged()

    # cmap_file = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68_C-14.cmap'
    # sc.read_cmap(cmap_file)

    # # Agglomative clustering
    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'
    # basename = f'{model_dir}/Atlases/{mname.split("/")[1]}'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # ea.export_map(Prob,atlas.name,cmap,labels,basename)

    # # Guided clustering
    # cluster_by = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-14'
    # Prob, parcel, atlas, labels, cmap = analyze_parcel(
    #     mname, sym=True, clustering='model_guided', cluster_by=cluster_by)
    # clustername = f'{model_dir}/Atlases/{mname.split("/")[1]}_C-{cluster_by.split("-")[-1]}'
    # ea.export_map(Prob, atlas.name, cmap, labels, clustername)

    # pass

    # # Plot fine, coarse and merged model
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_fine,sym=True)
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_coarse,sym=True)

    # # Show MNISymC2 Parcellation
    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_Kclus-40_meth-cosang'

    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-40'

    # output = f'{model_dir}/Atlases/{mname.split("/")[1]}'
    # ea.export_map(Prob, atlas.name, cmap, labels, output)

    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-80'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels, save_dir + '/exported/' + mname)

    # mname = 'Models_04/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-80'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels,save_dir + '/exported/' + mname)

    # --> Model 03, K=68

    # mname = 'Models_03/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-68'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels,save_dir + '/exported/' + mname)

    # mname = 'Models_04/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-68'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels,save_dir + '/exported/' + mname)

    # resample_atlas(mname)
    # make_asymmetry_map(mname)
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # cmap = mpl.cm.get_cmap('tab20')
    # rgb=cmap(np.arange(20))
    # plot_colormap(rgb)
    pass
