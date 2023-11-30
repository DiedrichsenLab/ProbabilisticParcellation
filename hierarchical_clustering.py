"""
Hierarchical Clustering

"""

import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
from Functional_Fusion.dataset import *
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as ev
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
# TEMPORARY FIX BEFORE MODELS ARE UPDATED TO HBP - REMOVE!!! TODO XX
import generativeMRF.arrangements as gar

import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.learn_fusion_gpu as lf

import torch as pt
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh, norm
from copy import deepcopy
import string


def parcel_similarity(model, plot=False, sym=False, weighting=None):
    """ Calculates a parcel similarity based on the V-vectors (functional profiles) of the emission models 

    Args:
        model (FullMultiModel): THe model 
        plot (bool, optional): Generate plot? Defaults to False.
        sym (bool, optional): Generate similarity in a symmetric fashion? Defaults to False.
        weighting (ndarray, optional): possible weighting of different dataset. Defaults to None.

    Returns:
        w_cos_sim: Weighted cosine similarity (integrated)
        cos_sim: Cosine similarity for each data set
        kappa: Kappa from each dataset(?)

    """
    n_sets = len(model.emissions)
    if sym:
        K = int(model.emissions[0].K / 2)
    else:
        K = model.emissions[0].K
    cos_sim = np.empty((n_sets, K, K))
    if model.emissions[0].uniform_kappa:
        kappa = np.empty((n_sets,))
    else:
        kappa = np.empty((n_sets, K))
    n_subj = np.empty((n_sets,))

    V = []
    for i, em in enumerate(model.emissions):
        if sym:
            # Average the two sides for clustering
            V.append(em.V[:, :K] + em.V[:, K:])
            V[-1] = V[-1] / np.sqrt((V[-1]**2).sum(axis=0))
            if model.emissions[0].uniform_kappa:
                kappa[i] = em.kappa
            else:
                kappa[i] = (em.kappa[:K] + em.kappa[K:]) / 2
        else:
            V.append(em.V)
            kappa[i] = em.kappa
        cos_sim[i] = V[-1].T @ V[-1]

        # V is weighted by Kappa and number of subjects
        V[-1] = V[-1] * np.sqrt(kappa[i] * em.num_subj)
        if weighting is not None:
            V[-1] = V[-1] * np.sqrt(weighting[i])

    # Combine all Vs and renormalize
    Vall = np.vstack(V)
    Vall = Vall / np.sqrt((Vall**2).sum(axis=0))
    # Calculate similarity
    w_cos_sim = Vall.T @ Vall

    # Integrated parcel similarity with kappa
    if plot is True:
        plt.figure()
        grid = int(np.ceil(np.sqrt(n_sets + 1)))
        for i in range(n_sets):
            plt.subplot(grid, grid, i + 1)
            plt.imshow(cos_sim[i, :, :], vmin=-1, vmax=1)
            plt.title(f"Dataset {i+1}")
        plt.subplot(grid, grid, n_sets + 1)
        plt.imshow(w_cos_sim, vmin=-1, vmax=1)
        plt.title(f"Merged")

    return w_cos_sim, cos_sim, kappa


def similarity_matrices(mname, sym=True):
    # Get model and atlas.
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = ut.load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.arrange.marginal_prob())
    index, cmap, labels = nt.read_lut(ut.model_dir + '/Atlases/' +
                                      fileparts[-1] + '.lut')
    K, P = Prob.shape
    if sym:
        K = int(K / 2)
        Prob = Prob[:K, :]
    labels = np.array(labels[1:K + 1])

    # Get parcel similarity:
    w_cos_sim, cos_sim, _ = parcel_similarity(model, plot=False, sym=sym)
    P = Prob / np.sqrt(np.sum(Prob**2, axis=1).reshape(-1, 1))

    spatial_sim = P @ P.T

    ind = np.argsort(labels)
    labels = labels[ind]
    w_cos_sim = w_cos_sim[:, ind][ind, :]
    spatial_sim = spatial_sim[:, ind][ind, :]
    return labels, w_cos_sim, spatial_sim, ind


def get_clusters(Z, K, num_cluster):
    cluster = np.zeros((K + Z.shape[0]), dtype=int)
    next_cluster = 1
    for i in np.arange(Z.shape[0] - num_cluster, -1, -1):
        indx = Z[i, 0:2].astype(int)
        # New cluster number
        if (cluster[i + K] == 0):
            cluster[i + K] = next_cluster
            cluster[indx] = next_cluster
            next_cluster += 1
        # Cluster already assigned - just pass down
        else:
            cluster[indx] = cluster[i + K]
    return cluster[:K], cluster[K:]


def agglomative_clustering(similarity,
                           sym=False,
                           num_clusters=5,
                           method='ward',
                           plot=True,
                           groups=['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G'],
                           cmap=None,
                           labels=None):
    # setting distance_threshold=0 ensures we compute the full tree.
    # plot the top three levels of the dendrogram
    K = similarity.shape[0]
    sym_sim = (similarity + similarity.T) / 2
    dist = squareform(1 - sym_sim.round(5))
    Z = linkage(dist, method)
    cleaves, clinks = get_clusters(Z, K, num_clusters)

    if plot:
        plt.figure()
        ax = plt.gca()

    # truncate_mode="level", p=3)
    R = dendrogram(Z, color_threshold=-1, no_plot=not plot)
    leaves = R['leaves']

    if labels is not None:
        if len(labels) != K:
            labels = labels[1:K + 1]
        # Remove l and r from labels
        labels = [label.strip('L').strip('R') for label in labels]
        # Order labels by clustering
        zipped_data = zip(labels, leaves)
        # Sort the zipped data based on positions
        sorted_data = sorted(zipped_data, key=lambda x: x[1])
        # Extract the sorted strings
        labels = [item[0] for item in sorted_data]
    else:
        # make the labels for the dendrogram
        labels = np.empty((K,), dtype=object)

        current = -1
        for i, l in enumerate(leaves):
            if cleaves[l] != current:
                num = 1
                current = cleaves[l]
            labels[i] = f"{groups[cleaves[l]]}{num}"
            num += 1

        # Make labels for mapping
        current = -1
        if sym:
            labels_map = np.empty((K * 2 + 1,), dtype=object)
            clusters = np.zeros((K * 2,), dtype=int)
            labels_map[0] = '0'
            for i, l in enumerate(leaves):
                if cleaves[l] != current:
                    num = 1
                    current = cleaves[l]
                labels_map[l + 1] = f"{groups[cleaves[l]]}{num}L"
                labels_map[l + K + 1] = f"{groups[cleaves[l]]}{num}R"
                clusters[l] = cleaves[l]
                clusters[l + K] = cleaves[l]
                num += 1
        else:
            labels_map = np.empty((K + 1,), dtype=object)
            clusters = np.zeros((K,), dtype=int)
            labels_map[0] = '0'
            for i, l in enumerate(leaves):
                labels_map[l + 1] = labels[i]
                clusters[l] = cleaves[l]

    if plot & (cmap is not None):
        ax.set_xticklabels(labels)
        ax.set_ylim((-0.2, 1.5))
        draw_cmap(ax, cmap, leaves, sym)
    return labels_map, clusters, leaves


def mixed_clustering(df_assignment, version=3):
    """ Maps parcels of a parcellation using a hand-coded merging of parcels
    specified in assignment.csv.

    Args:
        df_assignment: Assignment of fine parcels to coarse parcels for all parcels.

    Returns:
        fine_coarse_mapping: Assignment index  of fine parcels to coarse parcels
        fine_coarse_mapping_full: Assignment of fine parcels to coarse parcels for all parcels (same as fine_coarse_mapping for asym model)
    """
    # Get assignment of fine to medium parcels according to labels
    labels = df_assignment[f'labels_v{version}']
    labels = [label[:2] for label in labels]
    fine_coarse_mapping = [labels.index(label) for label in labels]
    
    # Create new labels for coarse map
    labels_new = [label + 'L' for label in labels]
    labels_new = list(dict.fromkeys(labels_new))
    labels_new = ['0'] + labels_new + [label.replace('L', 'R') for label in labels_new]
    return fine_coarse_mapping, labels_new


def draw_cmap(ax, cmap, leaves, sym, height=0.05):
    """ Draws the color map on the dendrogram"""
    K = len(leaves)
    for k in range(K):
        rect = Rectangle((k * 10, -height), 10, height,
                         facecolor=cmap(leaves[k] + 1),
                         fill=True,
                         edgecolor=(0, 0, 0, 1))
        ax.add_patch(rect)
    if sym:
        for k in range(K):
            # Left:
            rect = Rectangle((k * 10, -0.1), 10, 0.05,
                             facecolor=cmap(leaves[k] + 1 + K),
                             fill=True,
                             edgecolor=(0, 0, 0, 1))
            ax.add_patch(rect)


def make_symmetry_map(mname, cmap='hot', cscale=[0.3, 1]):
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = ut.load_batch_best(mname)

    # Get winner take-all
    Prob = np.array(model.marginal_prob())
    parcel = Prob.argmax(axis=0) + 1

    # Get similarity
    w_cos, _, _ = parcel_similarity(model, plot=False, sym=False)
    indx1 = np.arange(model.K)
    v = np.arange(model.arrange.K)
    indx2 = np.concatenate([v + model.arrange.K, v])
    sym_score = w_cos[indx1, indx2]

    suit_atlas, _ = am.get_atlas(split_mn[2].split(
        'space-')[1], ut.base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(parcel)
    surf_parcel = suit.flatmap.vol_to_surf(Nifti, stats='mode',
                                           space='MNISymC', ignore_zeros=True)
    surf_parcel = np.nan_to_num(surf_parcel, copy=False).astype(int)
    sym_map = np.zeros(surf_parcel.shape) * np.nan
    sym_map[surf_parcel > 0] = sym_score[surf_parcel[surf_parcel > 0] - 1]

    ax = suit.flatmap.plot(sym_map,
                           render='matplotlib',
                           overlay_type='func',
                           colorbar=True,
                           cscale=cscale,
                           cmap=cmap)
    # ax.show()
    return sym_score


def calc_parcel_size(Prob):
    """Calculates probabilstic and winner-take all cluster size 
    from probabilities 

    Args:
        Prob (ndarray): 
    returns: 
        sumP: sum of probabilities 
        sumV: number of hard-assigned voxels
    """
    if isinstance(Prob, pt.Tensor):
        Prob = Prob.numpy()
    if Prob.ndim == 3:
        voxel_axis = 2
    else:
        voxel_axis = 1
    parcel_axis = voxel_axis - 1
    sumP = np.sum(Prob, axis=voxel_axis)
    counts = np.zeros(Prob.shape)
    if Prob.ndim == 2:
        counts[np.argmax(Prob, axis=parcel_axis), np.arange(Prob.shape[voxel_axis])] = 1
    else:
        # Loop over subjects to get voxel counts for each subject
        for sub in np.arange(0, Prob.shape[0]):
            counts[sub, np.argmax(Prob, axis=parcel_axis)[sub], np.arange(Prob.shape[voxel_axis])] = 1
    sumV = np.sum(counts, axis=voxel_axis)
    return sumP, sumV

    


def plot_parcel_size(Prob, cmap, labels, wta=True, sort=True, side=None):
    sumP, sumV = calc_parcel_size(Prob)

    D = pd.DataFrame({'region': labels[1:],
                      'sumP': sumP,
                     'sumV': sumV,
                      'cnum': np.arange(Prob.shape[0]) + 1})
    if sort:
        D = D.sort_values(by='region')
    if side is not None:
        D['side'] = D.region.str[-1]
        D = D[D.side == side]
    pal = {d.region: cmap(d.cnum) for i, d in D.iterrows()}
    if wta:
        sb.barplot(data=D, y='region', x='sumV', palette=pal)
    else:
        sb.barplot(data=D, y='region', x='sumP', palette=pal)
    return D


def plot_parcel_mapping(fine_prob, coarse_prob, mapping, fine_labels=None):
    # get new probabilities
    ind = np.argsort(mapping)
    fine_prob = fine_prob[ind, :]
    mapping = mapping[ind]
    if not fine_labels is None:
        fine_labels = np.array(fine_labels)[ind]

    K1 = fine_prob.shape[0]
    K2 = coarse_prob.shape[0]
    indicator = np.zeros((K2, K1))
    indicator[mapping, np.arange(K1), ] = 1
    merged_prob = indicator @ fine_prob
    sumP = []
    sumV = []
    for p in [fine_prob, coarse_prob, merged_prob]:
        a, b = calc_parcel_size(p)
        sumP.append(a)
        sumV.append(b)

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(3)
    axs = gs.subplots(sharey=True)
    for i in range(3):
        K = len(sumP[i])
        axs[i].bar(np.arange(K), sumP[i])

    for i, s in enumerate(mapping):

        xyA = (i, 0)
        xyB = (mapping[i], 0)

        con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data",
                              coordsB="data", axesA=axs[0], axesB=axs[1], color="blue")
        axs[1].add_artist(con)
    pass
    if not fine_labels is None:
        axs[0].set_xticks(np.arange(K1))
        axs[0].set_xticklabels(fine_labels)


def guided_clustering(mname_fine, mname_coarse, method, fine_labels=None):
    """Maps parcels of a fine parcellation to parcels of a coarse parcellation guided by functional fusion model.

    Args:
        fine_probabilities: Probabilstic parcellation of a fine model (fine parcellation)
        coarse_probabilities: Probabilstic parcellation of a coarse model (coarse parcellation)

    Returns:
        fine_coarse_mapping: Winner-take-all assignment of fine parcels to coarse parcels
        fine_coarse_mapping_full: Winner-take-all assignment of fine parcels to coarse parcels for all parcels (same as fine_coarse_mapping for asym model)

    """
    # Import fine model
    fileparts = mname_fine.split('/')
    split_mn = fileparts[-1].split('_')
    _, fine_model = ut.load_batch_best(mname_fine)

    # Import coarse model
    fileparts = mname_coarse.split('/')
    split_mn = fileparts[-1].split('_')
    _, coarse_model = ut.load_batch_best(mname_coarse)

    # Get winner take all assignment for fine model
    fine_probabilities = pt.softmax(fine_model.arrange.logpi, dim=0)

    # Get probabilities of coarse model
    coarse_probabilities = pt.softmax(coarse_model.arrange.logpi, dim=0)

    fine_parcellation = fine_probabilities.argmax(axis=0)
    coarse_parcellation = coarse_probabilities.argmax(axis=0)

    print(
        f'\n Fine Model: \t\t{np.unique(fine_parcellation).shape[0]} WTA Parcels \n Coarse Model: \t\t{np.unique(coarse_parcellation).shape[0]} WTA Parcels')

    fine_coarse_mapping = np.zeros(fine_probabilities.shape[0], dtype=int)

    if method == 'hard':
        # For each fine parcel find the most probably coarse parcel
        # find voxels belonging to fine parcel
        for fine_parcel in np.unique(fine_parcellation):
            fine_voxels = (fine_parcellation == fine_parcel)
            # get probabilities of voxels belonging to each coarse parcel
            fine_coarse_prob = coarse_probabilities[:, fine_voxels]
            #  get winner take all assignment for mapping fine parcel to coarse parcel by adding within-fine-parcel voxel probabilities and assigning winner
            winner = fine_coarse_prob.sum(axis=1).argmax()
            # assign coarse parcel winner to fine parcel
            fine_coarse_mapping[fine_parcel] = winner.item()
    elif method == 'soft':
        cp = coarse_probabilities.sum(dim=1)
        fp = fine_probabilities.sum(dim=1, keepdim=True)
        overlap = fine_probabilities @ coarse_probabilities.T / cp
        fine_coarse_mapping = np.argmax(overlap.numpy(), axis=1)
    elif method == 'cosang':
        cp = pt.sum(coarse_probabilities**2, dim=1)
        fp = pt.sum(fine_probabilities**2, dim=1, keepdim=True)
        overlap = fine_probabilities @ coarse_probabilities.T / \
            pt.sqrt(cp * fp)
        fine_coarse_mapping = np.argmax(overlap.numpy(), axis=1)

    print(
        f'\n Clustered Model: \t{np.unique(fine_coarse_mapping).shape[0]} WTA Parcels \n')

    if type(fine_model.arrange) is ar.ArrangeIndependentSymmetric:
        fine_coarse_mapping_full = np.array(
            [*fine_coarse_mapping, *fine_coarse_mapping])
    else:
        fine_coarse_mapping_full = fine_coarse_mapping

    plot_parcel_mapping(fine_probabilities.numpy(),
                        coarse_probabilities.numpy(),
                        fine_coarse_mapping, fine_labels)

    return fine_coarse_mapping, fine_coarse_mapping_full


def cluster_labels(mapping, descriptor='alpha', sym=True):
    """Maps parcels of a fine parcellation to parcels of a coarse parcellation guided by functional fusion model.

    Args:
        mapping: Assignment of fine parcels to coarse parcels for all parcels.
                First half of mapping array MUST refer to left side parcels, second half MUST refer to right side parcels.
        descriptor: Cluster names ('alpha': Alphabetic cluster names)
        sym: Boolean indicating whether parcellation is symmetric

    Returns:
        labels: Region labels with naming convention <Letter><Number><Hemisphere>, i.e. A1L for parcel 1 in cluster A in left hemisphere.
        cluster_counts: Counts of regions within each cluster

    """
    # Move parcels up
    mapping = np.unique(
        mapping, return_inverse=True)[1]

    # labels = ['0']
    mapping_half = mapping[:int(mapping.shape[0] / 2)]

    if descriptor == 'alpha':
        groups = list(string.ascii_uppercase)[
            :len(np.unique(mapping_half))]

    K = np.unique(mapping_half).shape[0]

    # make the labels
    labels = np.empty((mapping_half.shape[0],), dtype=object)

    cluster_counts = [0] * len(groups)
    for i, l in enumerate(mapping_half):
        cluster_counts[l] = cluster_counts[l] + 1
        labels[i] = f"{groups[l]}{cluster_counts[l]}"

    # Make labels for mapping
    if sym:
        labels_left = labels + 'L'
        labels_right = labels + 'R'

        labels = labels_left.tolist() + labels_right.tolist()
        labels.insert(0, '0')
    else:
        raise (NotImplementedError('Asym labelling not yet implemented.'))

    return labels, cluster_counts


def merge_model(model, mapping):
    """Merges and reorders parcels from the an orginal fitted model. 

    Args:
        model:      Model to be clustered
        mapping ():    Cluster assignment for each model parcel

    Returns:
        new_model:  Clustered model
    """
    # Move parcels up
    mapping = np.unique(
        mapping, return_inverse=True)[1]

    # Get winner take all assignment for fine model
    Prob = pt.softmax(model.arrange.logpi, dim=0)

    # get new probabilities
    indicator = pcm.matrix.indicator(mapping)
    merged_probabilities = np.dot(indicator.T, (Prob))
    new_parcels = np.unique(mapping)

    # Create new, clustered model
    new_model = deepcopy(model)

    # Fill arrangement model parameteres
    new_model.arrange.logpi = pt.log(
        pt.tensor(merged_probabilities, dtype=pt.float32))
    new_model.arrange.set_param_list(['logpi'])
    new_model.arrange.K = int(len(new_parcels))

    if type(new_model.arrange) is gar.ArrangeIndependentSymmetric or type(new_model.arrange) is gar.ArrangeIndependentSeparateHem:
        all_parcels = [*new_parcels, *new_parcels + new_parcels.shape[0]]
        all_mappings = [*mapping, *mapping + new_parcels.shape[0]]
    else:
        all_parcels = new_parcels
        all_mappings = mapping

    new_model.arrange.K_full = len(all_parcels)

    # Fill emission model parameteres
    for e in np.arange(len(new_model.emissions)):
        new_model.emissions[e].K = int(len(all_parcels))

        # get new Vs
        V = new_model.emissions[e].V
        indicator = pcm.matrix.indicator(all_mappings)
        new_Vs = np.dot((V), indicator)
        new_model.emissions[e].V = pt.tensor(
            new_Vs, dtype=pt.get_default_dtype())
        new_model.emissions[e].set_param_list('V')

    return new_model


def cluster_parcel(mname_fine, fine_model=None, finfo=None, mname_new=None, f_assignment='assignment', version=3, refit_model=False, save_model=False):
    """Merges parcels of a fine probabilistic parcellation model into a reduced number of parcels using either guided or mixed clustering.

    Parameters:
    mname_fine(str): The file name of the fine probabilistic parcellation model.
    mname_new(str, optional): The file name for the merged probabilistic parcellation model. If not provided, the name will be constructed based on `mname_fine` and `method`. Defaults to None.
    f_assignment(str, optional): The file name of the mixed clustering assignment file to use. Defaults to 'assignment'.
    version(int, optional): The version of the mixed clustering assignment file to use. Defaults to 3.
    refit_model(bool, optional): Whether to refit the reduced model. Defaults to False.
    save_model(bool, optional): Whether to save the reduced model. Defaults to False.

    Returns:
    tuple: A tuple containing the reduced probabilistic parcellation model, the name of the reduced model, and the labels of the parcels in the reduced model.

    """
    # -- Import models --
    # Import fine model

    fileparts = mname_fine.split('/')
    split_mn = fileparts[-1].split('_')
    if fine_model is None:
        finfo, fine_model = ut.load_batch_best(mname_fine)
    finfo = ut.recover_info(finfo, fine_model, mname_fine)
    if split_mn[0] == 'sym':
        sym = True
    else:
        sym = False

    new_info = deepcopy(finfo)
    # Get mapping between fine parcels and coarse parcels
    df_assignment = pd.read_csv(
        ut.model_dir + '/Atlases/' + '/' + f_assignment + '.csv')
    mapping, labels = mixed_clustering(
        df_assignment, version=version)

    # -- Merge model --
    merged_model = merge_model(fine_model, mapping)

    # Add info
    new_info['model_type'] = mname_fine.split('/')[0]
    new_info['K_original'] = int(new_info.K)
    if sym:
        new_info['K'] = int(len(np.unique(mapping)) * 2)
    else:
        new_info['K'] = int(len(np.unique(mapping)))

    # Refit reduced model
    if refit_model:
        new_model, new_info = lf.refit_model(merged_model, new_info)
    else:
        new_model = merged_model
        new_info = pd.DataFrame(new_info.to_dict(), index=[0])

    # -- Save model --
    # Model is saved with K_coarse as cluster K, since using only the actual (effective) K might overwrite merged models stemming from different K_coarse
    if mname_new is None:
        mname_new = f'{mname_fine.split("_K-")[0]}_K-{new_info.K}_clustered'

    if save_model:
        # save new model
        with open(f'{ut.model_dir}/Models/{mname_new}.pickle', 'wb') as file:
            pickle.dump([new_model], file)

        # save new info
        new_info.to_csv(f'{ut.model_dir}/Models/{mname_new}.tsv',
                        sep='\t', index=False)

        print(
            f'Done. Saved merged model as: \n\t{mname_new} \nOutput folder: \n\t{ut.model_dir}/Models/ \n\n')

    return new_model, new_info, mname_new, labels




def reorder_model(
    mname,
    model=None,
    info=None,
    sym=True,
    mname_new=None,
    assignment="assignment.csv",
    version=3,
    save_model=False,
):
    """
    Reorders a saved parcellation model according to fixed order assignment.

    Args:
        mname (str): The name of the saved model to be reordered.
        model (object): The model object to be reordered. If None, the model will be loaded from the saved model file.
        info (object): The info that to be updated. If None, the info will be loaded from the saved model file. If given, it needs to be a pandas DataFrame or Series with one row.
        sym (bool): If True, reorders the model assuming symmetrical model.
        mname_new (str): The name of the reordered model. If None, the name will be the same as the original model with '_reordered' appended.
        assignment (str): The name of the CSV file containing the order assignment.
        version (int): Version of the model to create. Specifies the column name beginning with "idx_" in the assignment CSV file that contains the last ordering step to be concatenated with all previous reorderings to arrive at specified model version.
        save_model (bool): If True, saves the reordered model.

    Returns:
        new_model (object): The reordered model object.

    """
    # Get model and atlas.
    if model is None:
        info, model = ut.load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Get assignment
    assignment = pd.read_csv(f"{ut.model_dir}/Atlases/{assignment}")

    # Import indices up to idx
    order_arrange = np.array(assignment[f"idx_v0"])
    for i in np.arange(1, version+1):
        order_arrange = np.take(order_arrange, np.array(assignment[f"idx_v{i}"]))

    # Reorder the model
    new_model = deepcopy(model)
    if new_model.arrange.logpi.shape[0] == order_arrange.shape[0]:
        new_model.arrange.logpi = model.arrange.logpi[order_arrange]
    elif new_model.arrange.logpi.shape[0] * 2 == order_arrange.shape[0]:
        new_model.arrange.logpi = model.arrange.logpi[
            order_arrange[: len(order_arrange) // 2]
        ]
    elif new_model.arrange.logpi.shape[0] == np.unique(order_arrange).shape[0]:
        # Make order_arrange unique list of indices in the same order (necessary for re-ordering already merged models)
        order_arrange = order_arrange[
            np.sort(np.unique(order_arrange, return_index=True)[1])
        ]
        new_model.arrange.logpi = model.arrange.logpi[order_arrange]
    else:
        raise ValueError(
            "The number of parcels in the model does not match the number of parcels in the assignment."
        )

    order_emission = np.concatenate([order_arrange, order_arrange + len(order_arrange)])
    if not sym:
        order_arrange = order_emission

    for e, em in enumerate(new_model.emissions):
        new_model.emissions[e].V = em.V[:, order_emission]

    # Info
    new_info = deepcopy(info)
    new_info["ordered_by"] = version
    new_info = new_info.to_frame().T

    # Save the model
    if save_model:
        if mname_new is None:
            mname_new = mname + "_reordered"
        # save new model
        with open(f"{ut.model_dir}/Models/{mname_new}.pickle", "wb") as file:
            pickle.dump([new_model], file)

        # save new info
        new_info.to_csv(f"{ut.model_dir}/Models/{mname_new}.tsv", sep="\t", index=False)

        print(
            f"Done. Saved reordered model as: \n\t{mname_new} \nOutput folder: \n\t{ut.model_dir}/Models/ \n\n"
        )

    return new_model, new_info
