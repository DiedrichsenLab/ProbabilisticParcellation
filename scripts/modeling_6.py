#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional result 6 integrating resting state vs. purely task

Created on 2/17/2023 at 11:40 AM
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
import ProbabilisticParcellation.util as ut
from ProbabilisticParcellation.evaluate import *
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.hierarchical_clustering as cl
from ProbabilisticParcellation.learn_fusion_gpu import *

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
res_dir = model_dir + f'/Results' + '/5.all_datasets_fusion'


if __name__ == "__main__":
    K = 34
    sym_type = ['asym']
    model_type = ['03','04']
    space = 'MNISymC3'
    datasets_list = [1,2,3,4,5,6,7]

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    num_subj = T.return_nsubj.to_numpy()[datasets_list]
    sub_list = [np.arange(c) for c in num_subj[:-1]]

    # Odd indices for training, Even for testing
    hcp_train = np.arange(0, num_subj[-1], 2) + 1
    hcp_test = np.arange(0, num_subj[-1], 2)

    sub_list += [hcp_train]
    for t in model_type:
        for k in [10, 17, 20, 34, 40, 68, 100]:
            wdir, fname, info, models = fit_all(set_ind=datasets_list, K=k,
                                                repeats=100, model_type=t,
                                                sym_type=['asym'],
                                                subj_list=sub_list,
                                                space='MNISymC3')
            fname = fname + f'_hcpOdd'
            info.to_csv(wdir + fname + '.tsv', sep='\t')
            with open(wdir + fname + '.pickle', 'wb') as file:
                pickle.dump(models, file)
