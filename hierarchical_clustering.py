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
import PcmPy as pcm
from scipy.linalg import block_diag
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

atlas_dir = base_dir + '/Atlases'

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


def guided_clustering(fine_probabilities, coarse_probabilities):
    """Maps parcels of a fine parcellation to parcels of a coarse parcellation guided by functional fusion model.

    Args:
        fine_probabilities: Probabilstic parcellation of a fine model (fine parcellation)
        coarse_probabilities: Probabilstic parcellation of a coarse model (coarse parcellation)

    Returns:
        fine_coarse_mapping: Winner-take-all assignment of fine parcels to coarse parcels

    """
    

    fine_parcellation = fine_probabilities.argmax(axis=0)
    coarse_parcellation = coarse_probabilities.argmax(axis=0)

    print(f'---\n ++ Mapping values ++ \n Fine Model: \t{np.unique(fine_parcellation).shape[0]} WTA Parcels \n Coarse Model: \t{np.unique(coarse_parcellation).shape[0]} WTA Parcels')

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
    
    print(f'\n Clustered Model: \t{np.unique(fine_coarse_mapping).shape[0]} WTA Parcels \n')

    return fine_coarse_mapping


def merge_model(model, mapping):
    """Reduces model to effective K.

    Args:
        model:      Model to be clustered
        mapping:    Cluster assignment for each model parcel

    Returns:
        new_model:  Clustered model
    """
    # Move parcels up
    mapping_moved = np.unique(
        mapping, return_inverse=True)[1]
    
    # Get winner take all assignment for fine model
    Prob = pt.softmax(model.arrange.logpi, dim=0)

    # get new probabilities
    indicator = pcm.matrix.indicator(mapping_moved)
    merged_probabilities = np.dot(indicator.T, (Prob))
    new_parcels = np.unique(mapping_moved)

    # Create new, clustered model
    new_model = deepcopy(model)
    
    # Fill arrangement model parameteres
    new_model.arrange.logpi = pt.log(
        pt.tensor(merged_probabilities, dtype=pt.float32))
    new_model.arrange.set_param_list(['logpi'])
    new_model.arrange.K = int(len(new_parcels))
    
    if type(new_model.arrange) is ar.ArrangeIndependentSymmetric:
        all_parcels = [*new_parcels, *new_parcels]
    else:
        all_parcels = new_parcels
    
    new_model.arrange.K_full = len(all_parcels)

    # Fill emission model parameteres
    for e in np.arange(len(new_model.emissions)):
        new_model.emissions[e].K = int(len(all_parcels))
        new_model.emissions[e].V = new_model.emissions[e].V[:, all_parcels]
        new_model.emissions[e].set_param_list('V')
        
        # new_model.emissions[e].nparams = em.V.shape[0] * em.K
        # new_model.emissions[e].param_offset = [
        #     em.param_offset[0], em.K * em.V.shape[0], em.K * em.V.shape[0] + 1]
        # print(new_model.emissions[e].param_offset)
        # print(f'\n{[em.param_offset[0], em.K * em.V.shape[0], em.K * em.V.shape[0]+1]}\n\n')
        
    
    return new_model






