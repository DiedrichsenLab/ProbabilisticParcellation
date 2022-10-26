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

def plot_parcel_flat(parcel,cmap,atlas,render='matplotlib'):
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
    fig = suit.flatmap.plot(surf_data, 
                render=render,
                cmap=cmap, 
                new_figure=True,
                overlay_type='label')
    return fig

def parcel_similarity(model,plot=False,sym=False):
    n_sets = len(model.emissions)
    if sym:
        K = np.int(model.emissions[0].K/2)
    else:
        K = model.emissions[0].K
    cos_sim = np.empty((n_sets,K,K))
    kappa = np.empty((n_sets,))
    n_subj = np.empty((n_sets,))

    for i,em in enumerate(model.emissions):
        if sym:
            V = em.V[:,:K]+em.V[:,K:] # Average the two sides for clustering
            V = V/np.sqrt((V**2).sum(axis=0))
            cos_sim[i,:,:] = V.T @ V
        else:
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


def get_clusters(Z,K,num_cluster):
    cluster = np.zeros((K+Z.shape[0]))
    next_cluster = 1
    for i in range(Z.shape[0]):
        indx = Z[i,0:1]
        if np.all(cluster[indx]==0):
            

def agglomative_clustering(similarity,cmap,plot=True,sym=False):
    # setting distance_threshold=0 ensures we compute the full tree.
    # plot the top three levels of the dendrogram
    K = similarity.shape[0]
    sym_sim=(similarity+similarity.T)/2
    dist = squareform(1-sym_sim.round(5))
    Z = linkage(dist,'average')
    if plot:
        ax=plt.gca()
        R = dendrogram(Z) # truncate_mode="level", p=3)
        leaves = R['leaves']
        ax.set_ylim((-0.2,1.1))
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
        pass 

def colormap_mds(G,plot='2d',type='hsv'):
    N = G.shape[0]
    G = (G + G.T)/2
    Glam, V = eigh(G)
    Glam = np.flip(Glam,axis=0)
    V = np.flip(V,axis=1)

    # Make a new color maps 
    # Kill eigenvalues smaller than 3 
    # W=np.zeros((N,3))
    #ang = np.linspace(0,2*np.pi,N)
    # W[:,0]=np.cos(ang)
    # W[:,1]=np.sin(ang)
    W = V[:,:3] * np.sqrt(Glam[:3])
    if type=='rgb':
        rgb = (W-W.min())/(W.max()-W.min()) # Scale between zero and 1
    elif type=='hsv':
        # W=W-W.mean(axis=0)
        Sat=np.sqrt(W[:,0:1]**2)
        Sat = Sat/Sat.max()
        Hue=(np.arctan2(W[:,1],W[:,0])+np.pi)/(2*np.pi)
        Val = (W[:,2]-W[:,2].min())/(W[:,2].max()-W[:,2].min())*0.5+0.4
        rgb = mpl.colors.hsv_to_rgb(np.c_[Hue,Sat,Val])
    colors = np.c_[rgb,np.ones((N,))]
    colorsp = np.r_[np.zeros((1,4)),colors] # Add empty for the zero's color
    newcmp = ListedColormap(colorsp)

    if plot=='3d':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(W[:,0],W[:,1], W[:,2], marker='o',s=70,c=colors)
        ax.set_box_aspect((np.ptp(W[:,0]), np.ptp(W[:,1]), np.ptp(W[:,2]))) 
    if plot=='2d':
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(W[:,0],W[:,1], marker='o',s=70,c=colors)
        ax.set_aspect('equal','box') 
    return newcmp 


def analyze_parcel(mname,sym=True):
    split_mn = mname.split('_')
    info,model = load_batch_best(mname)

    # get the parcel similarity 
    w_cos_sim,_,_ = parcel_similarity(model,plot=False)

    # Make a colormap 
    cmap = colormap_mds(w_cos_sim,plot='3d',type='hsv')

    # Do clustering 
    plt.figure()
    w_cos_sym,_,_ = parcel_similarity(model,plot=False,sym=sym)
    agglomative_clustering(w_cos_sym,cmap,sym=sym)


    # Plot the parcellation 
    Prop = np.array(model.marginal_prob())
    parcel = Prop.argmax(axis=0)+1
    atlas = split_mn[2][6:]
    ax = plot_parcel_flat(parcel,cmap,atlas,render='plotly')
    
    pass



if __name__ == "__main__":
    mname = 'sym_MdPoNiIb_space-MNISymC3_K-34'
    analyze_parcel(mname)

    # agglomative_clustering(1-w_cos_sim)
