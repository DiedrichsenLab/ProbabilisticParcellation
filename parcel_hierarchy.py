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

pt.set_default_tensor_type(pt.FloatTensor)

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


def get_conditions(minfo):
    """Loads the conditions for a given dataset
    """

    datasets = minfo.datasets.strip("'[").strip("]'").split("' '")
    types = minfo.type.strip("'[").strip("]'").split("' '")
    sessions = minfo.sess.strip("'[").strip("]'").split("' '")
    conditions = []
    for i,dname in enumerate(datasets):
        _,dinfo,dataset = get_dataset(base_dir,dname,atlas=minfo.atlas,sess=sessions[i],type=types[i], info_only=True)
        condition_names = dinfo.drop_duplicates(subset=[dataset.cond_ind])
        condition_names = condition_names[dataset.cond_name].to_list()
        conditions.append([condition.split('  ')[0] for condition in condition_names])

    return conditions, datasets

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
    conditions, datasets = get_conditions(info)
    # (sanity check: profile length for each dataset should match length of condition list)
    # for i,cond in enumerate(conditions):
    #     print('Profile length matching n conditions {} :{}'.format(datasets[i],len(cond)==profile[i].shape[0]))

    return profile, conditions, datasets

def show_parcel_profile(p, profiles, conditions, datasets, show_ds='all', ncond=5, print=True):
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
            profile.append(dataset_profile)
            if print:
                print('{} :\t{}'.format(dataset, dataset_profile[:ncond]))

    else:
        # Collect condition names in order of parcel score from selected dataset
        d = datasets.index(show_ds)
        cond_name = conditions[d]
        cond_score = profiles[d][:,p].tolist()

        # sort conditions by condition score
        dataset_profile = [name for _,name in sorted(zip(cond_score,cond_name))]
        profile = dataset_profile
        if print:
            print('{} :\t{}'.format(datasets[d], dataset_profile[:ncond]))

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

def calc_mds(G,center=False):
    N = G.shape[0]
    if center:
        H = np.eye(N)-np.ones((N,N))/N
        G = H @ G @ H
    G = (G + G.T)/2
    Glam, V = eigh(G)
    Glam = np.flip(Glam,axis=0)
    V = np.flip(V,axis=1)
    W = V[:,:3] * np.sqrt(Glam[:3])

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

def make_orthonormal(U):
    """Gram-Schmidt process to make
    matrix orthonormal"""
    n = U.shape[1]
    V=U.copy()
    for i in range(n):
        prev_basis = V[:,0:i]     # orthonormal basis before V[i]
        rem = prev_basis @ prev_basis.T @ U[:,i]
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[:,i] = U[:,i] - rem
        V[:,i] /= norm(V[:,i])
    return V

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


def colormap_mds(W,target=None,clusters=None,gamma=0.3):
    """Map the similarity structure of MDS to a colormap
    Args:
        W (ndarray): N x 3 array of original multidimensional scaling
        target (tuple): Target origin [0] directions[1] of the desired map
        clusters (ndarray): distorts color towards cluster mean
        gamma (float): Strength of cluster mean
    Returns:
        colormap (Listed Colormap):
    """
    N = W.shape[0]
    if target is not None:
        tm=target[0]
        tV = target[1]

        # Get the eigenvalues of W around the origin.
        m=np.mean(W[:,:3],axis=0)
        A=W-m
        # Get the eigenvalues in ascending order
        l,V=eigh(A.T@A)
        l = np.flip(l,axis=0)
        V = np.flip(V,axis=1)
        # Rotate and shift the color space towards the target
        Wm = A @ V @ tV.T
        Wm += tm
    # rgb = (W-W.min())/(W.max()-W.min()) # Scale between zero and 1
    Wm[Wm<0]=0
    Wm[Wm>1]=1
    if clusters is not None:
        M = np.zeros((clusters.max(),3))
        for i in np.unique(clusters):
            M[i-1,:]=np.mean(Wm[clusters==i,:],axis=0)
            Wm[clusters==i,:]=(1-gamma) * Wm[clusters==i,:] + gamma * M[i-1]

    colors = np.c_[Wm,np.ones((N,))]
    colorsp = np.r_[np.zeros((1,4)),colors] # Add empty for the zero's color
    newcmp = ListedColormap(colorsp)
    return newcmp

def export_map(data,atlas,cmap,labels,base_name):
    """Exports a new atlas map as a Nifti (probseg), Nifti (desg), Gifti, and lut-file.

    Args:
        data (probabilities): Probabilstic parcellation to export
        atlas (): FunctionalFusion atlas type (SUIT2,MNISym3)
        cmap (): Colormap
        labels (list): List of labels for fields
        base_name (_type_): File basename for atlas
    """
    # Transform cmap into numpy array
    if not isinstance(cmap,np.ndarray):
        cmap = cmap(np.arange(cmap.N))


    suit_atlas, _ = am.get_atlas(atlas,base_dir + '/Atlases')
    probseg = suit_atlas.data_to_nifti(data)
    parcel = np.argmax(data,axis=0)+1
    dseg = suit_atlas.data_to_nifti(parcel)

    # Figure out correct mapping space
    if atlas[0:4]=='SUIT':
        map_space='SUIT'
    elif atlas[0:7]=='MNISymC':
        map_space='MNISymC'
    else:
        raise(NameError('Unknown atlas space'))

    # Plotting label
    surf_data = suit.flatmap.vol_to_surf(probseg, stats='nanmean',
            space=map_space)
    surf_parcel = np.argmax(surf_data,axis=1)+1
    Gifti = nt.make_label_gifti(surf_parcel.reshape(-1,1),
                anatomical_struct='Cerebellum',
                labels = np.arange(surf_parcel.max()+1),
                label_names=labels,
                label_RGBA = cmap)

    nb.save(dseg,base_name + f'_space-{atlas}_dseg.nii')
    nb.save(probseg,base_name + f'_space-{atlas}_probseg.nii')
    nb.save(Gifti,base_name + '_dseg.label.gii')
    save_lut(np.arange(len(labels)),cmap[:,0:4],labels, base_name + '.lut')

def save_lut(index,colors,labels,fname):
    L=pd.DataFrame({
            "key":index,
            "R":colors[:,0].round(4),
            "G":colors[:,1].round(4),
            "B":colors[:,2].round(4),
            "Name":labels})
    L.to_csv(fname,header=None,sep=' ',index=False)


def renormalize_probseg(probseg):
    X = probseg.get_fdata()
    xs = np.sum(X,axis=3)
    xs[xs<0.5]=np.nan
    X = X/np.expand_dims(xs,3)
    X[np.isnan(X)]=0
    probseg_img = nb.Nifti1Image(X,probseg.affine)
    parcel = np.argmax(X,axis=3)+1
    parcel[np.isnan(xs)]=0
    dseg_img = nb.Nifti1Image(parcel.astype(np.int8),probseg.affine)
    dseg_img.set_data_dtype('int8')
    # dseg_img.header.set_intent(1002,(),"")
    probseg_img.set_data_dtype('float32')
    # probseg_img.header.set_slope_inter(1/(2**16-1),0.0)
    return probseg_img,dseg_img

def resample_atlas(base_name):
    """ Resamples probabilstic atlas into 1mm resolution and
    SUIT space
    """
    mnisym_dir=base_dir + '/Atlases/tpl-MNI152NLin2000cSymC'
    suit_dir=base_dir + '/Atlases/tpl-SUIT'

    # Reslice to 1mm MNI and 1mm SUIT space
    print('reslicing to 1mm')
    sym3 = nb.load(mnisym_dir + f'/{base_name}_space-MNISymC3_probseg.nii')
    tmp1 = nb.load(mnisym_dir + f'/tpl-MNISymC_res-1_gmcmask.nii')
    shap = tmp1.shape+sym3.shape[3:]
    sym1 = ns.resample_from_to(sym3,(shap,tmp1.affine),3)
    print('normalizing')
    sym1,dsym1= renormalize_probseg(sym1)
    print('saving')
    nb.save(sym1,mnisym_dir + f'/{base_name}_space-MNISymC_probseg.nii')
    nb.save(dsym1,mnisym_dir + f'/{base_name}_space-MNISymC_dseg.nii')

    # Now put the image into SUIT space
    print('reslicing to SUIT')
    deform = nb.load(mnisym_dir + '/tpl-MNI152NLin2009cSymC_space-SUIT_xfm.nii')
    suit1 = nt.deform_image(sym1,deform,1)
    print('normalizing')
    suit1,dsuit1= renormalize_probseg(suit1)
    print('saving')
    nb.save(suit1,suit_dir + f'/{base_name}_space-SUIT_probseg.nii')
    nb.save(dsuit1,suit_dir + f'/{base_name}_space-SUIT_dseg.nii')

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

def reduce_model(new_model, new_parcels):
    """Reduces model to effective K.

    Args:
        new_model: Coarse model containing voxel probabilities of fine model (Clustered model)
        new_parcels: Vector of new parcels

    Returns:
        reduced_model: Reduced model with empty parcels removed (Clustered fine model with effective K)

    """
    reduced_model = deepcopy(new_model)

    reduced_model.arrange.logpi = new_model.arrange.logpi[new_parcels]

    if hasattr(new_model, 'K_sym'):
        reduced_model.K_sym = int(len(new_parcels))
        all_parcels = [*new_parcels, *new_parcels + new_model.K_sym]
    else:
        all_parcels = new_parcels

    reduced_model.K = int(len(all_parcels))
    reduced_model.arrange.K = int(len(all_parcels))

    # Reduce emission models
    for e, em in enumerate(new_model.emissions):
        reduced_model.emissions[e].V = em.V[:, all_parcels]
        reduced_model.emissions[e].K = int(len(all_parcels))


    # Reduce emission model K and kappa
    if hasattr(new_model, 'K_sym'):
        for e, em in enumerate(new_model.emissions):
            if not em.uniform_kappa:
                raise NotImplementedError('Reducing of nonuniform kappa models not implemented yet.')
                # reduced_model.emissions[e] = em.V[:, all_parcels]

    return reduced_model

def cluster_model(mname_fine, mname_coarse, sym=True, reduce=True):
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

    # If merged parcels are fewer than coarse parcels, create reduced model
    if len(new_K_sym) < coarse_model.arrange.K and reduce:
        new_model = reduce_model(new_model, new_parcels)

    K = len(new_K_sym)*2
    return new_model, K


def get_clustered_model(mname_fine, mname_coarse, sym=True, reduce=True):
    """Merges the parcels of a fine parcellation model according to a coarser model.

    Args:
        mname_fine: Probabilstic parcellation to merge (fine parcellation)
        mname_caorse: Probabilstic parcellation that determines how to merge (coarse parcellation)
        merge: Vector that indicates cluster assignment

    """

    # Import fine model
    fileparts = mname_fine.split('/')
    split_mn = fileparts[-1].split('_')
    finfo,fmodel = load_batch_best(mname_fine)

    # Import coarse model
    fileparts = mname_coarse.split('/')
    split_mn = fileparts[-1].split('_')
    cinfo,cmodel = load_batch_best(mname_coarse)

    merged_model, K = merge_model(fmodel, cmodel, reduce=reduce)
    mname_merged = f'{mname_fine}_merged_K-{K}'

    # -- Save model --    
    # save new model
    with open(f'{model_dir}/Models/{mname_merged}.pickle', 'wb') as file:
        pickle.dump(new_model, file)

    # save new info
    finfo['K_original'] = int(finfo.K)
    finfo['K'] = int(K_new)
    finfo['K_coarse'] = int(K_coarse)
    finfo.to_csv(f'{model_dir}/Models/{mname_merged}.tsv', sep='\t')

    return new_model, mname_merged, mapping


# def get_clustered_model(mname_fine, mname_coarse, sym=True, reduce=True):
#     """Merges the parcels of a fine parcellation model according to a coarser model.

#     Args:
#         mname_fine: Probabilstic parcellation to merge (fine parcellation)
#         mname_caorse: Probabilstic parcellation that determines how to merge (coarse parcellation)
#         merge: Vector that indicates cluster assignment
        
#     """
    
#     # Import fine model
#     fileparts = mname_fine.split('/')
#     split_mn = fileparts[-1].split('_')
#     finfo,fmodel = load_batch_best(mname_fine)

#     # Import coarse model
#     fileparts = mname_coarse.split('/')
#     split_mn = fileparts[-1].split('_')
#     cinfo,cmodel = load_batch_best(mname_coarse)

#     K_coarse = split_mn[-1].split('-')[1]
#     merged_model, K_new, mapping = merge_model(fmodel, cmodel, reduce=reduce)
#     mname_merged = f'{mname_fine}_merged-{K_coarse}'

#     # save new model
#     with open(f'{model_dir}/Models/{mname_merged}.pickle', 'wb') as file:
#         pickle.dump(merged_model, file)

#     # save new info
#     finfo['K_original'] = int(finfo.K)
#     finfo['K'] = int(K_new)
#     finfo['K_coarse'] = int(K_coarse)
#     finfo.to_csv(f'{model_dir}/Models/{mname_merged}.tsv', sep='\t')

#     return merged_model, mname_merged


def analyze_parcel(mname, load_best=True, sym=True):

    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    if load_best:
        info,model = load_batch_best(mname)
    else:
        info,model = load_batch_fit(mname)

    # Get parcel similarity:
    w_cos_sym,_,_ = parcel_similarity(model,plot=True,sym=sym)

    # groups=['I','L','W','A','O','D','M']
    # Do Clustering:
    num_clusters = 7
    if num_clusters > model.arrange.K:
        num_clusters = model.arrange.K-1
    labels,clusters,leaves = agglomative_clustering(w_cos_sym,sym=sym,num_clusters=4,plot=False)
    ax = plt.gca()

    # Make a colormap
    w_cos_sim,_,_ = parcel_similarity(model,plot=False)
    W = calc_mds(w_cos_sim,center=True)

    # Define color anchors
    m = np.array([0.65,0.65,0.65])

    # Desired orientation of the eigenvectors of MDS in colorspace
    V=np.array([[-0.3,-0.6,1],[1,-.6,-.7],[1,1,1]]).T
    V=make_orthonormal(V)

    cmap = colormap_mds(W,target=(m,V),clusters=clusters,gamma=0)

    # Replot the Clustering dendrogram, this time with the correct color map
    agglomative_clustering(w_cos_sym,sym=sym,num_clusters=7,plot=True,cmap=cmap)
    plot_colormap(cmap(np.arange(model.K)))

    # Plot the parcellation
    Prob = np.array(model.marginal_prob())
    parcel = Prob.argmax(axis=0)+1
    atlas = split_mn[2][6:]
    ax = plot_data_flat(parcel,atlas,cmap = cmap,
                    dtype='label',
                    labels=labels,
                    render='plotly')
    ax.show()

    return Prob,parcel,atlas,labels,cmap

def make_sfn_atlas():
    Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # Quick hack - hard-code the labels:
    labels = np.array(['0', 'O1L', 'W1L', 'A2L', 'A3L', 'L1L', 'O2L', 'D1L', 'L2L', 'M2L','I1L', 'D2L', 'M3L', 'M4L', 'M1L', 'W4L', 'A1L', 'W2L', 'O1R', 'W1R', 'A2R', 'A3R', 'L1R', 'O2R', 'D1R', 'L2R', 'M2R', 'I1R', 'D2R', 'M3R', 'M4R', 'M1R', 'W4R', 'A1R', 'W2R'], dtype=object)
    export_map(Prob,atlas,cmap,labels,base_dir + '/Atlases/tpl-MNI152NLin2000cSymC/atl-NettekovenSym34')
    resample_atlas('atl-NettekovenSym34')



def merge_clusters():
    save_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/Models/'
    # --- Merge parcels at K=20, 34 & 40 ---
    merged_models = []
    # for k in [10, 14, 20, 28, 34, 40]:
    for k in [14]:

        mname_fine = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'
        mname_coarse = f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-{k}'

        # merge model
        _, mname_merged, mapping = cluster_model(mname_fine, mname_coarse, sym=True, reduce=True)
        merged_models.append(mname_merged)



if __name__ == "__main__":
    
    mname = 'Models_03/sym_MdPo_space-MNISymC3_K-40'
    Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    pass


    # export the merged model
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_merged, load_best=False, sym=True)
    # export_map(Prob,atlas,cmap,labels, save_dir + '/exported/' + mname_merged)

    # # Plot fine, coarse and merged model
    Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_fine,sym=True)
    Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_coarse,sym=True)

    # --- Show Merged Parcellation at K=20, K=34, K=40---
    # mname_fine = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_fine,sym=True)
    # for mname_merged in merged_models:
    #     Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_merged, load_best=False, sym=True)


    # # Show MNISymC2 Parcellation
    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-10'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels,save_dir + '/exported/' + mname)

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

