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
from util import *
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

def parcel_similarity(model,plot=False,sym=False, weighting=None):
    n_sets = len(model.emissions)
    if sym:
        K = np.int(model.emissions[0].K/2)
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
        else:
            V.append(em.V)
        cos_sim[i]=V[-1].T @ V[-1]

        # V is weighted by Kappa and number of subjects
        kappa[i] = em.kappa
        V[-1] = V[-1] * np.sqrt(em.kappa * em.num_subj)
        if weighting is not None:
            V[-1] = V[-1] * np.sqrt(weighting[i])

    # Combine all Vs and renormalize
    Vall = np.vstack(V)
    Vall = Vall/np.sqrt((Vall**2).sum(axis=0))
    w_cos_sim = Vall.T @ Vall

    # Integrated parcel similarity with kappa
    if plot is True:
        grid = int(np.ceil(np.sqrt(n_sets+1)))
        for i in range(n_sets):
            plt.subplot(grid,grid,i+1)
            plt.imshow(cos_sim[i,:,:],vmin=-1,vmax=1)
            plt.title(f"Dataset {i+1}")
        plt.subplot(grid,grid,n_sets+1)
        plt.imshow(w_cos_sim,vmin=-1,vmax=1)
        plt.title(f"Merged")

    return w_cos_sim,cos_sim,kappa


def get_conditions(minfo):
    """Loads the conditions for a given dataset
    """

    datasets = minfo.datasets.strip("'[").strip("]'").split("' '")
    types = minfo.type.strip("'[").strip("]'").split("' '")
    sessions = minfo.sess.strip("'[").strip("]'").split("' '")
    conditions = []
    for i,dname in enumerate(datasets):
        _,dinfo,dataset = get_dataset(base_dir,dname,atlas=minfo.atlas,sess=sessions[i],type=types[i])
        condition_names = dinfo.drop_duplicates(subset=[dataset.cond_ind])
        condition_names = condition_names[dataset.cond_name].to_list()
        conditions.append([condition.split('  ')[0] for condition in condition_names])

    return conditions

def get_profiles(model,info):
    """Returns the functional profile for each parcel
    Args:
        model: Loaded model
        info: Model info
    Returns:
        profile: V for each emission model
        conditions: list of condition lists for each dataset
    """
    profile = [em.V for em in model.emissions]
    # load the condition for each dataset
    conditions = get_conditions(info)
    # (sanity check: profile length for each dataset should match length of condition list)
    # for i,cond in enumerate(conditions):
    #     print('Profile length matching n conditions {} :{}'.format(datasets[i],len(cond)==profile[i].shape[0]))

    return profile, conditions

def show_parcel_profile(p, profiles, conditions, datasets, show_ds='all', ncond=5):
    """Returns the functional profile for a given parcel either for selected dataset or all datasets
    Args:
        profiles: parcel scores for each condition in each dataset
        conditions: condition names of each dataset
        datasets: dataset names
        show_ds: selected dataset
                'Mdtb'
                'Pontine'
                'Nishimoto'
                'Ibc'
                'Hcp'
                'all'
        ncond: number of highest scoring conditions to show

    Returns:
        profile: condition names in order of parcel score

    """
    if show_ds =='all':
        # Collect condition names in order of parcel score from all datasets
        profile = []
        for d,dataset in enumerate(datasets):
            cond_name = conditions[d]
            cond_score = profiles[d][:,p].tolist()
            # sort conditions by condition score
            dataset_profile = [name for _,name in sorted(zip(cond_score,cond_name),reverse=True)]
            print('{} :\t{}'.format(dataset, dataset_profile[:ncond]))
            profile.append(dataset_profile)

    else:
        # Collect condition names in order of parcel score from selected dataset
        d = datasets.index(show_ds)
        cond_name = conditions[d]
        cond_score = profiles[d][:,p].tolist()

        # sort conditions by condition score
        dataset_profile = [name for _,name in sorted(zip(cond_score,cond_name))]
        print('{} :\t{}'.format(datasets[d], dataset_profile[:ncond]))
        profile = dataset_profile

    return profile

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

def agglomative_clustering(similarity,cmap,
                        plot=True,
                        sym=False,
                        num_clusters=5,
                        method = 'ward'):
    # setting distance_threshold=0 ensures we compute the full tree.
    # plot the top three levels of the dendrogram
    K = similarity.shape[0]
    sym_sim=(similarity+similarity.T)/2
    dist = squareform(1-sym_sim.round(5))
    Z = linkage(dist,method)
    cleaves,clinks = get_clusters(Z,K,num_clusters)

    # Determine colors
    if sym:
        colr = (cmap(np.arange(K)+1) + cmap(np.arange(K)+ K +1))/2
    else:
        colr = cmap(np.arange(K)+1)
    group_color=np.empty((num_clusters+1,4))
    for i in np.unique(cleaves):
        group_color[i,:]=colr[cleaves==i,:].mean(axis=0)
    link_colors = group_color[clinks,:]

    ax=plt.gca()
    R = dendrogram(Z,color_threshold=-1) # truncate_mode="level", p=3)
    leaves = R['leaves']
    # make the labels for the dendrogram
    groups = ['A','B','C','D','E','F','G']
    labels = np.empty((K,),dtype=object)

    current = -1
    for i,l in enumerate(leaves):
        if cleaves[l]!=current:
            num=1
            current = cleaves[l]
        labels[i]=f"{groups[cleaves[l]]}{num}"
        num+=1
    ax.set_xticklabels(labels)

    # Make labels for mapping
    current = -1
    if sym:
        labels_map = np.empty((K*2+1,),dtype=object)
        labels_map[0] = '0'
        for i,l in enumerate(leaves):
            if cleaves[l]!=current:
                num=1
                current = cleaves[l]
            labels_map[l+1]   = f"{groups[cleaves[l]]}{num}L"
            labels_map[l+K+1] = f"{groups[cleaves[l]]}{num}R"
            num+=1
    else:
        labels_map = np.empty((K+1,),dtype=object)
        labels_map[0] = '0'
        for i,l in enumerate(leaves):
            labels_map[l+1]   = labels[i]

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
    return labels_map


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
        Sat=np.sqrt(W[:,0:1]**2)
        Sat = Sat/Sat.max()
        Hue=(np.arctan2(W[:,1],W[:,0])+np.pi)/(2*np.pi)
        Val = (W[:,2]-W[:,2].min())/(W[:,2].max()-W[:,2].min())*0.5+0.4
        rgb = mpl.colors.hsv_to_rgb(np.c_[Hue,Sat,Val])
    elif type=='hsv2':
        Sat=np.sqrt(V[:,0:1]**2)
        Sat = Sat/Sat.max()
        Hue=(np.arctan2(V[:,1],V[:,0])+np.pi)/(2*np.pi)
        Val = (V[:,2]-V[:,2].min())/(V[:,2].max()-V[:,2].min())*0.5+0.4
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
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info,model = load_batch_best(mname)

    # get the parcel similarity
    w_cos_sim,_,_ = parcel_similarity(model,plot=True)

    # Make a colormap
    cmap = colormap_mds(w_cos_sim,plot='3d',type='rgb')

    # Do clustering
    plt.figure()
    w_cos_sym,_,_ = parcel_similarity(model,plot=False,sym=sym)
    labels = agglomative_clustering(w_cos_sym,cmap,sym=sym)

    # Plot the parcellation
    Prop = np.array(model.marginal_prob())
    parcel = Prop.argmax(axis=0)+1
    atlas = split_mn[2][6:]

    ax = plot_data_flat(parcel,atlas,cmap = cmap,
                    dtype='label',
                    labels=labels,
                    render='plotly')
    ax.show()
    pass



if __name__ == "__main__":
    mname = 'Models_04/sym_MdPoNiIb_space-MNISymC3_K-34'
    analyze_parcel(mname,sym=True)
    pass

