"""
Hierarchical Clustering

"""

import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
from Functional_Fusion.dataset import *
import generativeMRF.emissions as em
import generativeMRF.arrangements as ar
import generativeMRF.full_model as fm
import generativeMRF.evaluation as ev
from scipy.linalg import block_diag
import PcmPy as pcm
import nibabel as nb
import nibabel.processing as ns
import SUITPy as suit
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import pickle
from ProbabilisticParcellation.util import *
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh, norm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from copy import deepcopy

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(base_dir).exists():
    base_dir = '/Users/jdiedrichsen/Data/FunctionalFusion/'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))


def parcel_similarity(model,plot=False,sym=False, weighting=None):
    n_sets = len(model.emissions)
    if sym:
        K = int(model.emissions[0].K/2)
    else:
        K = model.emissions[0].K
    cos_sim = np.empty((n_sets,K,K))
    if model.emissions[0].uniform_kappa:
        kappa = np.empty((n_sets,))
    else:
        kappa = np.empty((n_sets,K))
    n_subj = np.empty((n_sets,))

    V = []
    for i,em in enumerate(model.emissions):
        if sym:
            V.append(em.V[:,:K]+em.V[:,K:]) # Average the two sides for clustering
            V[-1] = V[-1]/np.sqrt((V[-1]**2).sum(axis=0))
            if model.emissions[0].uniform_kappa:
                kappa[i] = em.kappa
            else:
                kappa[i] = (em.kappa[:K]+em.kappa[K:])/2
        else:
            V.append(em.V)
            kappa[i] = em.kappa
        cos_sim[i]=V[-1].T @ V[-1]

        # V is weighted by Kappa and number of subjects
        V[-1] = V[-1] * np.sqrt(kappa[i] * em.num_subj)
        if weighting is not None:
            V[-1] = V[-1] * np.sqrt(weighting[i])

    # Combine all Vs and renormalize
    Vall = np.vstack(V)
    Vall = Vall/np.sqrt((Vall**2).sum(axis=0))
    w_cos_sim = Vall.T @ Vall

    # Integrated parcel similarity with kappa
    if plot is True:
        plt.figure()
        grid = int(np.ceil(np.sqrt(n_sets+1)))
        for i in range(n_sets):
            plt.subplot(grid,grid,i+1)
            plt.imshow(cos_sim[i,:,:],vmin=-1,vmax=1)
            plt.title(f"Dataset {i+1}")
        plt.subplot(grid,grid,n_sets+1)
        plt.imshow(w_cos_sim,vmin=-1,vmax=1)
        plt.title(f"Merged")

    return w_cos_sim,cos_sim,kappa


def get_clusters(Z,K,num_cluster):
    cluster = np.zeros((K+Z.shape[0]),dtype=int)
    next_cluster = 1
    for i in np.arange(Z.shape[0]-num_cluster,-1,-1):
        indx = Z[i,0:2].astype(int)
        # New cluster number
        if (cluster[i+K]==0):
            cluster[i+K]  = next_cluster
            cluster[indx] = next_cluster
            next_cluster += 1
        # Cluster already assigned - just pass down
        else:
            cluster[indx]=cluster[i+K]
    return cluster[:K],cluster[K:]

def agglomative_clustering(similarity,
                        sym=False,
                        num_clusters=5,
                        method = 'ward',
                        plot=True,
                        groups = ['0','A','B','C','D','E','F','G'],
                        cmap=None):
    # setting distance_threshold=0 ensures we compute the full tree.
    # plot the top three levels of the dendrogram
    K = similarity.shape[0]
    sym_sim=(similarity+similarity.T)/2
    dist = squareform(1-sym_sim.round(5))
    Z = linkage(dist,method)
    cleaves,clinks = get_clusters(Z,K,num_clusters)

    if plot:
        plt.figure()
        ax=plt.gca()

    R = dendrogram(Z,color_threshold=-1,no_plot=not plot) # truncate_mode="level", p=3)
    leaves = R['leaves']
    # make the labels for the dendrogram
    labels = np.empty((K,),dtype=object)

    current = -1
    for i,l in enumerate(leaves):
        if cleaves[l]!=current:
            num=1
            current = cleaves[l]
        labels[i]=f"{groups[cleaves[l]]}{num}"
        num+=1

    # Make labels for mapping
    current = -1
    if sym:
        labels_map = np.empty((K*2+1,),dtype=object)
        clusters = np.zeros((K*2,),dtype=int)
        labels_map[0] = '0'
        for i,l in enumerate(leaves):
            if cleaves[l]!=current:
                num=1
                current = cleaves[l]
            labels_map[l+1]   = f"{groups[cleaves[l]]}{num}L"
            labels_map[l+K+1] = f"{groups[cleaves[l]]}{num}R"
            clusters[l] = cleaves[l]
            clusters[l+K] = cleaves[l]
            num+=1
    else:
        labels_map = np.empty((K+1,),dtype=object)
        clusters = np.zeros((K,),dtype=int)
        labels_map[0] = '0'
        for i,l in enumerate(leaves):
            labels_map[l+1]   = labels[i]
            clusters[l] = cleaves[l]
    if plot & (cmap is not None):
        ax.set_xticklabels(labels)
        ax.set_ylim((-0.2,1.5))
        draw_cmap(ax,cmap,leaves,sym)
    return labels_map,clusters,leaves

def draw_cmap(ax,cmap,leaves,sym):
    """ Draws the color map on the dendrogram"""
    K = len(leaves)
    for k in range(K):
        rect = Rectangle((k*10, -0.05), 10,0.05,
        facecolor=cmap(leaves[k]+1),
        fill=True,
        edgecolor=(0,0,0,1))
        ax.add_patch(rect)
    if sym:
        for k in range(K):
            # Left:
            rect = Rectangle((k*10, -0.1), 10,0.05,
            facecolor=cmap(leaves[k]+1+K),
            fill=True,
            edgecolor=(0,0,0,1))
            ax.add_patch(rect)

def make_asymmetry_map(mname, cmap='hot', cscale=[0.3,1]):
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info,model = load_batch_best(mname)

    # Get winner take-all
    Prob = np.array(model.marginal_prob())
    parcel = Prob.argmax(axis=0)+1

    # Get similarity
    w_cos,_,_ = parcel_similarity(model,plot=False,sym=False)
    indx1 = np.arange(model.K)
    v = np.arange(model.K_sym)
    indx2 = np.concatenate([v+model.K_sym,v])
    sym_score = w_cos[indx1,indx2]

    suit_atlas, _ = am.get_atlas('MNISymC3',base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(parcel)
    surf_parcel = suit.flatmap.vol_to_surf(Nifti, stats='mode',
            space='MNISymC',ignore_zeros=True)
    surf_parcel = np.nan_to_num(surf_parcel,copy=False).astype(int)
    sym_map = np.zeros(surf_parcel.shape)*np.nan
    sym_map[surf_parcel>0] = sym_score[surf_parcel[surf_parcel>0]-1]

    ax = suit.flatmap.plot(sym_map,
                render='matplotlib',
                overlay_type='func',
                colorbar=True,
                cscale=cscale,
                cmap=cmap)
    # ax.show()
    return sym_score

def map_fine2coarse(fine_probabilities, coarse_probabilities):
    """Maps parcels of a fine parcellation to parcels of a coarse parcellation.

    Args:
        fine_probabilities: Probabilstic parcellation of a fine model (fine parcellation)
        coarse_probabilities: Probabilstic parcellation of a coarse model (coarse parcellation)

    Returns:
        fine_coarse_mapping: Winner-take-all assignment of fine parcels to coarse parcels

    """
    print(f'--- Assigning {mname_fine.split("/")[1]} to {mname_coarse.split("/")[1]} ---\n ++ Start values ++ \n Fine Model: \t{fine_probabilities.shape[0]} Prob Parcels \n Coarse Model: \t{coarse_probabilities.shape[0]} Prob Parcels')

    fine_parcellation = fine_probabilities.argmax(axis=0)
    coarse_parcellation = coarse_probabilities.argmax(axis=0)

    print(f'\n Fine Model: \t{np.unique(fine_parcellation).shape[0]} WTA Parcels \n Coarse Model: \t{np.unique(coarse_parcellation).shape[0]} WTA Parcels')

    fine_coarse_mapping = np.zeros(fine_probabilities.shape[0])
    for fine_parcel in (fine_parcellation).unique():
        # find voxels belonging to fine parcel
        fine_voxels = (fine_parcellation == fine_parcel)
        # get probabilities of voxels belonging to each coarse parcel
        fine_coarse_prob = coarse_probabilities[:,fine_voxels]
        # get winner take all assignment for mapping fine parcel to coarse parcel by adding within-fine-parcel voxel probabilities and assigning winner
        winner = fine_coarse_prob.sum(axis=1).argmax()
        # assign coarse parcel winner to fine parcel
        fine_coarse_mapping[fine_parcel] = winner.item()
    
    print(f'\n Merged Model: \t{np.unique(fine_coarse_mapping).shape[0]} WTA Parcels \n')
    return fine_coarse_mapping

def reduce_model(new_model, new_info, new_parcels):
    """Reduces model to effective K.

    Args:
        new_model:      Coarse model containing voxel probabilities of fine model (Clustered model)
        new_info:       Information for new model (Clustered model)
        new_parcels:    Vector of new parcels

    Returns:
        new_model: Reduced model with empty parcels removed (Clustered fine model with effective K)

    """
    new_model = deepcopy(new_model)

    if hasattr(new_model, 'K_sym'):
        new_model.K_sym = int(len(new_parcels))
        all_parcels = [*new_parcels, *new_parcels]
    else:
        all_parcels = new_parcels
        
    new_model.arrange.K = int(len(new_parcels))
    new_model.arrange.nparams = new_model.P * new_model.arrange.K
    new_model.arrange.logpi = new_model.arrange.logpi[new_parcels]

    # Refit emission models
    print(f'Freezing arrangement model and fitting emission models...')    

    for e, em in enumerate(new_model.emissions):
        # new_model.emissions[e].V = em.V[:, all_parcels]
        new_model.emissions[e].K = int(len(all_parcels))
        new_model.emissions[e].nparams = em.V.shape[0] * em.K

    if hasattr(new_model, 'K_sym'):
        atlas, _ = am.get_atlas(new_info.atlas, atlas_dir, sym=True)
        M = fm.FullMultiModelSymmetric(new_model.arrange, new_model.emissions,
                                       atlas.indx_full, atlas.indx_reduced,
                                       same_parcels=False)
    else:
        M = fm.FullMultiModel(ar_model, em_models)

    
    model_settings = {'Models_01': [True, True, False],
                      'Models_02': [False, True, False],
                      'Models_03': [True, False, False],
                      'Models_04': [False, False, False],
                      'Models_05': [False, True, True]}
    
    uniform_kappa = model_settings[new_info.model_type][0]
    join_sess = model_settings[new_info.model_type][1]
    join_sess_part = model_settings[new_info.model_type][2]

    datasets = new_info.datasets.strip("'[").strip("]'").split("' '")
    sessions = new_info.sess.strip("'[").strip("]'").split("' '")
    types = new_info.type.strip("'[").strip("]'").split("' '")

    data, cond_vec, part_vec, subj_ind = build_data_list(datasets,
                                                         atlas=new_info.atlas,
                                                         sess=sessions,
                                                         type=types,
                                                         join_sess=join_sess,
                                                         join_sess_part=join_sess_part)

    # Copy the object (without data)
    m = deepcopy(M)
    # Attach the data
    m.initialize(data, subj_ind=subj_ind)

    m, ll, theta, U_hat, ll_init = m.fit_em(
            iter=1,
            tol=0.01,
            fit_emission=True,
            fit_arrangement=False)

    return m

def cluster_model(mname_fine, mname_coarse, sym=True):
    """Merges the parcels of a fine parcellation model according to a coarser model.

    Args:
        mname_fine:     Probabilstic parcellation to merge (fine parcellation)
        mname_caorse:   Probabilstic parcellation that determines how to merge (coarse parcellation)
        sym:            Boolean indicating if model is symmetric. Defaults to True.
        reduce:         Boolean indicating if model should be reduced (empty parcels removed). Defaults to True.

    Returns:
        merged_model:   Merged model. Coarse model containing voxel probabilities of fine model (Clustered fine model)
        mname_merged:   Name of merged model
        mapping:        Mapping of fine parcels to coarse parcels.

    """
    # -- Import models --    
    # Import fine model
    fileparts = mname_fine.split('/')
    split_mn = fileparts[-1].split('_')
    finfo,fine_model = load_batch_best(mname_fine)

    # Import coarse model
    fileparts = mname_coarse.split('/')
    split_mn = fileparts[-1].split('_')
    cinfo,coarse_model = load_batch_best(mname_coarse)

    # -- Cluster fine model --    
    K_coarse = split_mn[-1].split('-')[1]
    mname_merged = f'{mname_fine}_merged-{K_coarse}'

    # Get winner take all assignment for fine model
    fine_probabilities = pt.softmax(fine_model.arrange.logpi,dim=0)

    # Get probabilities of coarse model
    coarse_probabilities = pt.softmax(coarse_model.arrange.logpi,dim=0)
    
    # Get mapping between fine parcels and coarse parcels
    mapping = map_fine2coarse(fine_probabilities, coarse_probabilities)
    new_K_sym = np.unique(mapping)

    # -- Make new model --    
    # Initiliaze new probabilities
    new_probabilities = np.zeros(coarse_probabilities.shape)

    # # Initiliaze new probabilities with small prob instead of zero
    # min_val = float("1e-30")
    # new_probabilities = np.repeat(min_val, coarse_probabilities.flatten().shape).reshape(coarse_probabilities.shape)

    # get new probabilities
    indicator = pcm.matrix.indicator(mapping)
    merged_probabilities = np.dot(indicator.T, (fine_probabilities))

    # sort probabilities according to original coarse parcels
    new_parcels = [int(k) for k in new_K_sym]
    merged_probabilities_sorted = np.array([x for _,x in sorted(zip(new_parcels,merged_probabilities))])
    new_probabilities[new_parcels] = merged_probabilities_sorted
    
    # Create new, clustered model
    new_model = deepcopy(coarse_model)    

    # fill new probabilities
    new_model.arrange.logpi = pt.log(pt.tensor(new_probabilities, dtype=pt.float32))

    # Make new info
    K_new = len(new_K_sym)*2
    new_info = deepcopy(finfo)
    new_info['K_original'] = int(finfo.K)
    new_info['K'] = int(K_new)
    new_info['K_coarse'] = int(K_coarse)
    new_info['model_type'] = mname_fine.split('/')[0]

    # Create reduced model
    new_model = reduce_model(new_model, new_info, new_parcels)

    # -- Save model --    
    # save new model
    with open(f'{model_dir}/Models/{mname_merged}.pickle', 'wb') as file:
        pickle.dump([new_model], file)

    # save new info
    # TODO: Make sure this is a dataframe, not a series
    new_info.to_csv(f'{model_dir}/Models/{mname_merged}.tsv', sep='\t')

    return new_model, mname_merged, mapping




