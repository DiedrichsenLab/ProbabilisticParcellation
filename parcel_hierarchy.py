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
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))


def parcel_similarity(model_name,plot=False):
    info,models,Prop,V = load_batch_fit(model_name)
    j=np.argmax(info.loglik)
    m = models[j]
    n_sets = len(m.emissions)
    cos_sim = np.empty((n_sets,m.K,m.K))
    kappa = np.empty((n_sets,))
    n_subj = np.empty((n_sets,))
    for i,em in enumerate(m.emissions):
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

def agglomative_clustering(similarity):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, 
                                    n_clusters=None,
                                    affinity='precomputed')

    model = model.fit(X)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()





if __name__ == "__main__":
    mname = 'sym_MdPoNiIb_space-MNISymC3_K-20'
    parcel_similarity(mname,plot=True)
    pass


