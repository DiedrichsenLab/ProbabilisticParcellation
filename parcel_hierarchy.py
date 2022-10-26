"""Build a hierarchie of parcels from a parcelation
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
import nibabel as nb
import SUITPy as suit
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import pickle
from evaluate import *
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))


def load_batch_best(fname):
    """ Loads a batch of model fits and selects the best one
    Args:
        fname (str): File name
    """
    wdir = base_dir + '/Models/'
    info = pd.read_csv(wdir + fname + '.tsv',sep='\t')
    with open(wdir + fname + '.pickle','rb') as file:
        models = pickle.load(file)
    j = info.loglik.argmax()
    return info.iloc[j],models[j]

def get_parcel(model):
    Prop = np.array(model.arrange.marginal_prob())
    if hasattr(model,'P_sym'):
        Prop_full = np.zeros((model.K,model.P))
        Prop_full[:model.K_sym,model.indx_full[0]]=Prop
        Prop_full[model.K_sym:,model.indx_full[1]]=Prop
        Prop=Prop_full
    parcel = Prop.argmax(axis=0)+1
    return parcel

def plot_parcel_flat(parcel,cmap,atlas):
    # Plot Parcellation 
    suit_atlas = am.get_atlas(atlas,base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(parcel)
    if atlas[0:4]=='SUIT':
        map_space='SUIT'
    elif atlas[0:7]=='MNISymC':
        map_space='MNISymC'
    else:
        raise(NameError('Unknown atlas space'))

    surf_data = suit.flatmap.vol_to_surf(Nifti, stats='mode',
            space=map_space,ignore_zeros=True)
    suit.flatmap.plot(surf_data, 
                render='matplotlib',
                cmap=cmap, 
                new_figure=False,
                overlay_type='label')


def parcel_similarity(model,plot=False):
    n_sets = len(model.emissions)
    K = model.emissions[0].K
    cos_sim = np.empty((n_sets,K,K))
    kappa = np.empty((n_sets,))
    n_subj = np.empty((n_sets,))
    for i,em in enumerate(model.emissions):
        cos_sim[i,:,:] = em.V.T @ em.V
        kappa[i] = em.kappa
        n_subj[i] = em.num_subj
    # Integrated parcel similarity with kappa
    weight = kappa * n_subj
    w_cos_sim = (cos_sim * weight.reshape((-1,1,1))).sum(axis=0)/weight.sum()
    if plot is True:
        for i in range(n_sets):
            plt.subplot(1,n_sets+1,i+1)
            plt.imshow(cos_sim[i,:,:],vmin=-1,vmax=1)
        plt.subplot(1,n_sets+1,n_sets+1)
        plt.imshow(w_cos_sim,vmin=-1,vmax=1)

    return w_cos_sim,cos_sim,kappa

def agglomative_clustering(similarity,cmap,plot=True):
    # setting distance_threshold=0 ensures we compute the full tree.
    # plot the top three levels of the dendrogram
    K = similarity.shape[0]
    sym_sim=(similarity+similarity.T)/2
    dist = squareform(1-sym_sim.round(5))
    Z = linkage(dist,'average')
    if plot:
        ax=plt.gca()
        R = dendrogram(Z) # truncate_mode="level", p=3)
        ax.set_ylim((-0.2,1.1))
        leaves = R['leaves']
        for k in range(K):
            rect = Rectangle((k*10, -0.05), 10,0.05,
            facecolor=cmap(leaves[k]+1),
            fill=True,
            edgecolor=(0,0,0,1))
            ax.add_patch(rect)
        pass 

def colormap_mds(G,plot=True):
    N = G.shape[0]
    G = (G + G.T)/2
    Glam, V = eigh(G)
    Glam = np.flip(Glam,axis=0)
    V = np.flip(V,axis=1)

    # Make a new color maps 
    # Kill eigenvalues smaller than 3 
    W = V[:,:3] * np.sqrt(Glam[:3])
    sW = (W-W.min())/(W.max()-W.min())
    colors = np.c_[sW,np.ones((N,))]
    colorsp = np.r_[np.zeros((1,4)),colors] # Add empty for the zero's color
    newcmp = ListedColormap(colorsp)


    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(W[:,0],W[:,1], W[:,2], marker='o',
                   c=colors)
        ax.set_box_aspect((np.ptp(W[:,0]), np.ptp(W[:,1]), np.ptp(W[:,2]))) 
    return newcmp 


def analyze_parcel(mname):
    split_mn = mname.split('_')
    info,model = load_batch_best(mname)

    # get the parcel similarity 
    w_cos_sim,_,_ = parcel_similarity(model,plot=False)

    # Make a colormap 
    cmap = colormap_mds(w_cos_sim)

    # plt.figure()
    # parcel = get_parcel(model)
    # atlas = split_mn[2][6:]
    # plot_parcel_flat(parcel,cmap,atlas)
    plt.figure()
    agglomative_clustering(w_cos_sim,cmap)
    pass



if __name__ == "__main__":
    mname = 'sym_Md_space-MNISymC3_K-34'
    analyze_parcel(mname)

    # agglomative_clustering(1-w_cos_sim)
