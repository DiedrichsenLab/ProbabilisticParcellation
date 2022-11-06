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

def agglomative_clustering(similarity,
                        sym=False,
                        num_clusters=5,
                        method = 'ward',
                        plot=True,
                        cmap=None):
    # setting distance_threshold=0 ensures we compute the full tree.
    # plot the top three levels of the dendrogram
    K = similarity.shape[0]
    sym_sim=(similarity+similarity.T)/2
    dist = squareform(1-sym_sim.round(5))
    Z = linkage(dist,method)
    cleaves,clinks = get_clusters(Z,K,num_clusters)

    ax=plt.gca()
    R = dendrogram(Z,color_threshold=-1) # truncate_mode="level", p=3)
    leaves = R['leaves']
    # make the labels for the dendrogram
    groups = ['0','A','B','C','D','E','F','G']
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
    ax.set_ylim((-0.2,1.5))
    if cmap is not None:
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


def calc_mds(G,clusters=None,gamma = 0.2,center=False):
    N = G.shape[0]
    if center:
        H = np.eye(N)-np.ones((N,N))/N
        G = H @ G @ H
    G = (G + G.T)/2
    Glam, V = eigh(G)
    Glam = np.flip(Glam,axis=0)
    V = np.flip(V,axis=1)
    W = V[:,:3] * np.sqrt(Glam[:3])

    if clusters is not None:
        M = np.zeros((clusters.max(),3))
        for i in np.unique(clusters):
            M[i-1,:]=np.mean(W[clusters==i,:],axis=0)
            W[clusters==i,:]=(1-gamma) * W[clusters==i,:] + gamma * M[i-1]
    return W

"""elif type=='hsv':
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
    elif type=='rgb_cluster':

        rgb = (W-W.min())/(W.max()-W.min()) # Scale between zero and 1
    else:
        raise(NameError(f'Unknown Type: {type}'))
"""

def get_target(cmap):
    if isinstance(cmap,str):
        cmap = mpl.cm.get_cmap(cmap)
    rgb=cmap(np.arange(cmap.N))
    # plot_colormap(rgb)
    tm=np.mean(rgb[:,:3],axis=0)
    A=rgb[:,:3]-tm
    tl,tV=eigh(A.T@A)
    tl = np.flip(tl,axis=0)
    tV = np.flip(tV,axis=1)
    return tm,tl,tV

def plot_colormap(rgb):
    N,a = rgb.shape
    if a==3:
        rgb = np.c_[rgb,np.ones((N,))]
    rgba = np.r_[rgb,[[0,0,0,1],[1,1,1,1]]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rgba[:,0],rgba[:,1], rgba[:,2], marker='o',s=70,c=rgba)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    m=np.mean(rgb[:,:3],axis=0)
    A=rgb[:,:3]-m
    l,V=eigh(A.T@A)
    l = np.flip(l,axis=0)
    V = np.flip(V,axis=1)

    B = V * np.sqrt(l) * 0.5
    for i in range(2):
        ax.quiver(m[0],m[1],m[2],B[0,i],B[1,i],B[2,i])
    return m,l,V


def colormap_mds(W,target=None,scale=False):
    """Map the simularity structure of MDS to a colormap
    Args:
        W (_type_): _description_
        plot (str, optional): _description_. Defaults to '2d'.
        type (str, optional): _description_. Defaults to 'hsv'.
        target (stg or )
    Returns:
        colormap: _description_
    """
    N = W.shape[0]
    if target is not None:
        tm=target[0]
        tl=target[1]
        tV = target[2]
        m=np.mean(W[:,:3],axis=0)
        A=W-m
        l,V=eigh(A.T@A)
        l = np.flip(l,axis=0)
        V = np.flip(V,axis=1)
        Wm = A @ V @ tV.T
        Wm += tm
    # rgb = (W-W.min())/(W.max()-W.min()) # Scale between zero and 1
    Wm[Wm<0]=0
    Wm[Wm>1]=1
    colors = np.c_[Wm,np.ones((N,))]
    colorsp = np.r_[np.zeros((1,4)),colors] # Add empty for the zero's color
    newcmp = ListedColormap(colorsp)
    return newcmp

def analyze_parcel(mname,sym=True):
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info,model = load_batch_best(mname)

    # Do clustering
    w_cos_sym,_,_ = parcel_similarity(model,plot=False,sym=sym)
    labels,clusters,leaves = agglomative_clustering(w_cos_sym,sym=sym,num_clusters=5)
    ax = plt.gca()

    # Make a colormap
    # get the parcel similarity
    w_cos_sim,_,_ = parcel_similarity(model,plot=True)
    W = calc_mds(w_cos_sim,clusters=clusters,gamma=0.3,center=True)
    m,l,V = get_target('tab20')
    V[:,0]=V[:,0]*-1
    cmap = colormap_mds(W,target=(m,l,V))
    draw_cmap(ax,cmap,leaves,sym)
    plot_colormap(cmap(np.arange(34)))

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
    # cmap = mpl.cm.get_cmap('tab20')
    # rgb=cmap(np.arange(20))
    # plot_colormap(rgb)
    pass

