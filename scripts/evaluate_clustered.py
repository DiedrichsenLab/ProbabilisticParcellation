#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripts for functional fusion model paper ONLY!
It may not capatible with other major functions, use with caution!

Created on 12/13/2022 at 12:39 PM
Author: dzhi
"""
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
import time
import pickle
from copy import copy,deepcopy
from itertools import combinations
from ProbabilisticParcellation.util import *
from ProbabilisticParcellation.evaluate import *


# pytorch cuda global flag
# pt.cuda.is_available = lambda : False
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# Find model directory to save model fitting results
model_dir = 'Y:/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    raise (NameError('Could not find model_dir'))

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:/data/FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))

atlas_dir = base_dir + f'/Atlases'
res_dir = model_dir + f'/Results'

def evaluate(K=10, symmetric='asym', model_type=None, model_name=None,
                  t_datasets=None, return_df=False, k_merged=None, load_best=True):
    """Evaluate group and individual DCBC and coserr of all dataset fusion
       and any dataset training standalone on each of the datasets.
    Args:
        K: the number of parcels
        t_datasets (list): a list of test datasets
    Returns:
        Write in evaluation file
    """
    # Preparing model type, name, and test set is not given
    if model_type is None:
        model_type = ['01','02','03','04','05']

    if model_name is None:
        model_name = ['Md','Po','Ni','Ib','Wm','De','So','MdPoNiIbWmDeSo']

    if t_datasets is None:
        t_datasets = ['MDTB','Pontine','Nishimoto','IBC',
                      'WMFS','Demand','Somatotopic']

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    m_name = []
    for t in model_type:
        print(f'- Start evaluating Model_{t} - {model_name}...')
        m_name += [f'Models_{t}/{symmetric}_{nam}_space-MNISymC3_K-{K}' for nam in model_name]
        if k_merged is not None:
            m_name = [f'Models_{t}/{symmetric}_{nam}_space-MNISymC3_K-{K}_merged_K-{k_merged}' for nam in model_name]
            load_best=False

    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        # 1. Run DCBC individual
        res_dcbc = run_dcbc_individual(m_name, ds, 'all', cond_ind=None,
                                       part_ind='half', indivtrain_ind='half',
                                       indivtrain_values=[1,2], device='cuda', load_best=load_best)
        # 2. Run coserr individual
        res_coserr = run_prederror(m_name, ds, 'all', cond_ind=None,
                                   part_ind='half', eval_types=['group', 'floor'],
                                   indivtrain_ind='half', indivtrain_values=[1,2],
                                   device='cuda', load_best=load_best)
        # 3. Merge the two dataframe
        res = pd.merge(res_dcbc, res_coserr, how='outer')
        results = pd.concat([results, res], ignore_index=True)

    if return_df:
        return results
    else:
        # Save file
        wdir = model_dir + f'/Models/Evaluation'
        fname = f'/eval_all_{symmetric}_K-{K}_datasetFusion.tsv'
        if k_merged is not None:
            fname = f'/eval_all_{symmetric}_K-{K}__merged_K-{k_merged}.tsv'
        results.to_csv(wdir + fname, index=False, sep='\t')


if __name__ == "__main__":
    
    ############# Result 6: Clustered models #############
    T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    D = pd.DataFrame()
    datasets = [0, 1, 2, 3, 4, 5, 6]
    datanames = T.two_letter_code[datasets].to_list()
    
    for i in range(7):        
        for k_merged in [10,18,22,26]:
            res = evaluate(K=68, symmetric='sym', model_type=['03'],
                                model_name=['MdPoNiIbWmDeSo'], t_datasets=[T.name[i]],
                                return_df=True, k_merged=k_merged)
            D = pd.concat([D, res], ignore_index=True)
    wdir = model_dir + f'/Models/Evaluation/sym'
    fname = f'/eval_all_sym_MdPoNiIbWmDeSo_merged_teston_indivDataset.tsv'
    D.to_csv(wdir + fname, index=False, sep='\t')

    pass