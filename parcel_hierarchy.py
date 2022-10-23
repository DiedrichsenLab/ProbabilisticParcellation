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
    for i,em in enumerate(m.emissions):
        cos_sim[i,:,:] = em.V.T @ em.V
        kappa[i] = em.kappa
    # Integrated parcel similarity with kappa
    w_cos_sim = (cos_sim * kappa.reshape((-1,1,1))).sum(axis=0)/kappa.sum()
    if plot is True:
        for i in range(n_sets):
            plt.subplot(1,n_sets+1,i+1)
            plt.imshow(cos_sim[i,:,:],vmin=-1,vmax=1)
        plt.subplot(1,n_sets+1,n_sets+1)
        plt.imshow(w_cos_sim,vmin=-1,vmax=1)

    return w_cos_sim,cos_sim,kappa


if __name__ == "__main__":
    mname = 'sym_MdPoNiIb_space-MNISymC3_K-20'
    parcel_similarity(mname,plot=True)
    pass


