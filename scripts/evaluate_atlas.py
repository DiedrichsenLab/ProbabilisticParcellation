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
import Functional_Fusion.dataset as ds
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
from copy import copy, deepcopy
from itertools import combinations
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.evaluate as ev


res_dir = ut.model_dir + f'/Results/nettekoven_68'

'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68.tsv'
'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed.tsv'


def evaluation(model_name, test_datasets):
    # determine device:

    results = pd.DataFrame()
    for dset in test_datasets:
        print(f'Testdata: {dset}\n')

        # Preparing atlas, cond_vec, part_vec
        atlas, _ = am.get_atlas('MNISymC2', atlas_dir=ut.base_dir + '/Atlases')
        tdata, tinfo, tds = ds.get_dataset(
            ut.base_dir, dset, atlas='MNISymC2', sess='all')
        # default from dataset class
        cond_vec = tinfo[tds.cond_ind].values.reshape(-1, )
        part_vec = tinfo['half'].values
        # part_vec = np.ones((tinfo.shape[0],), dtype=int)
        CV_setting = [('half', 1), ('half', 2)]

        ################ CV starts here ################
        for (indivtrain_ind, indivtrain_values) in CV_setting:
            # get train/test index for cross validation
            train_indx = tinfo[indivtrain_ind] == indivtrain_values
            test_indx = tinfo[indivtrain_ind] != indivtrain_values
            # 1. Run DCBC individual
            res_dcbc = ev.run_dcbc(model_name, tdata, atlas,
                                   train_indx=train_indx,
                                   test_indx=test_indx,
                                   cond_vec=cond_vec,
                                   part_vec=part_vec,
                                   device=ut.default_device)
            res_dcbc['indivtrain_ind'] = indivtrain_ind
            res_dcbc['indivtrain_val'] = indivtrain_values
            res_dcbc['test_data'] = dset

            results = pd.concat([results, res_dcbc], ignore_index=True)
    return results


def evaluate_sym(K=[10, 14, 20, 28, 34, 40, 48, 56, 60, 68], train_type=['indiv', 'loo', 'all'], test_datasets=['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                                                                                                                'WMFS', 'Demand', 'Somatotopic']):
    """Evaluate models that were fitted in MNISymC2 space on all datasets
    """
    pass


def evaluate_clustered(test_datasets=['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                                      'WMFS', 'Demand', 'Somatotopic', 'HCP']):
    """Evalute models that were clustered according to mixed method.
    """

    model_name = f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'

    # Evaluate
    results = evaluation(model_name, test_datasets)

    # Save file
    fname = 'eval_' + model_name.split(' / ')[-1] + '.tsv'
    results.to_csv(res_dir + fname, index=False, sep='\t')


if __name__ == "__main__":
    evaluate_clustered()

    pass
