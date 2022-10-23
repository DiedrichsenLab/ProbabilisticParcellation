# Script for evaluating DCBC on fused parcellation
from time import gmtime
from pathlib import Path
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
import DCBC.DCBC_vol as dcbc

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))

def eval_generative_SNMF(model_names = ['asym_Md_space-SUIT3_K-10']):
    """This is the evaluation case of the parcellation comparison
    between the new fusion model vs. convex semi non-negative matrix
    factorization (King et. al, 2019).

    Args:
        model_names (list): the list of model names to be evaluated
    Returns:
        the plot
    Note:
        I'm just simply curious whether the fusion model fit on mdtb
        standalone (indepent ar. + VMF em.) can beat NMF algorithm
        or not. So nothing hurt the script.    -- dzhi
    """
    # Use specific mask / atlas.
    mask = base_dir + '/Atlases/tpl-SUIT/tpl-SUIT_res-3_gmcmask.nii'
    atlas = am.AtlasVolumetric('SUIT3', mask_img=mask)

    # get original mdtb parcels (nmf)
    from learn_mdtb import get_mdtb_parcel
    mdtb_par, _ = get_mdtb_parcel(do_plot=False)
    mdtb_par = np.where(mdtb_par == 0, np.nan, mdtb_par)

    parcel = np.empty((len(model_names), atlas.P))

    for i, mn in enumerate(model_names):
        info, models, Prop, V = load_batch_fit(mn)
        j = np.argmax(info.loglik)
        # Get winner take all
        par = pt.argmax(Prop[j, :, :], dim=0) + 1
        parcel[i, :] = np.where(np.isnan(mdtb_par), np.nan, par.numpy())

    # Evaluate case: use all MDTB data
    # It kinda of overfitting but still fair comparison
    data_eval, _, _ = get_all_mdtb(atlas='SUIT3')
    dcbc_base = eval_dcbc(mdtb_par, atlas, func_data=data_eval,
                          resolution=3, trim_nan=True)

    dcbc_compare = []
    for p in range(parcel.shape[0]):
        this_dcbc = eval_dcbc(parcel[p], atlas, func_data=data_eval,
                              resolution=3, trim_nan=True)
        dcbc_compare.append(this_dcbc)

    plt.figure()
    plt.bar(['NMF', 'ind+vmf'], [dcbc_base.mean(), dcbc_compare[0].mean()],
            yerr=[dcbc_base.std() / np.sqrt(24),
                  dcbc_compare[0].std() / np.sqrt(24)])
    plt.show()

