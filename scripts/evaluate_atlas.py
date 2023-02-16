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
from datetime import datetime


res_dir = ut.model_dir + f'/Models/Evaluation/nettekoven_68/'


def evaluation(model_name, test_datasets):
    # determine space:
    space = model_name.split('space-')[-1].split('_')[0]

    results = pd.DataFrame()
    for dset in test_datasets:
        print(f'Testdata: {dset}\n')

        # Preparing atlas, cond_vec, part_vec
        atlas, _ = am.get_atlas(space, atlas_dir=ut.base_dir + '/Atlases')
        tdata, tinfo, tds = ds.get_dataset(
            ut.base_dir, dset, atlas=space, sess='all')
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


def evaluate_sym(K=[68], train_type=['indiv', 'loo', 'all'], rest_included=False, out_file=None):
    """Evaluate models that were fitted in MNISymC2 space on all datasets
    """
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    datasets_long = T['name'].tolist()
    datasets_short = T['two_letter_code'].tolist()

    if rest_included:
        datasets = datasets_short
        test_datasets = datasets_long
    else:
        datasets = datasets_short[:-1]
        test_datasets = datasets_long[:-1]

    # Get model names for models to evaluate
    indiv = datasets
    leave_one_out = ["".join(datasets_short[:i] + datasets[i + 1:])
                     for i in range(len(datasets))]
    all = ''.join(datasets)
    train_types = {'indiv': datasets, 'loo': leave_one_out, 'all': [all]}
    train_datasets = []
    for tt in train_type:
        train_datasets.extend(train_types[tt])

    if rest_included:
        train_datasets.extend(['Hc'])

    model_name = [f'Models_03/sym_{train_dset}_space-MNISymC3_K-{this_k}'
                  for this_k in K for train_dset in train_datasets]

    # Evaluate
    Results = pd.DataFrame()
    for mname in model_name:
        results = evaluation(mname, test_datasets)
        Results = pd.concat([Results, results], ignore_index=True)

    # Save file
    if out_file is None:
        timestamp = datetime.today().strftime('%Y-%m-%d')
        out_file = 'eval_' + timestamp + '.tsv'
    results.to_csv(res_dir + out_file, index=False, sep='\t')

    pass


def evaluate_clustered(test_datasets=['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                                      'WMFS', 'Demand', 'Somatotopic', 'HCP']):
    """Evalute models that were clustered according to mixed method.
    """

    model_name = [
        'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed']

    # Evaluate
    results = evaluation(model_name, test_datasets)

    # Save file
    fname = 'eval_' + model_name.split(' / ')[-1] + '.tsv'
    results.to_csv(res_dir + fname, index=False, sep='\t')


def evaluate_selected(test_datasets=['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                                     'WMFS', 'Demand', 'Somatotopic', 'HCP']):
    """Evalute models that were clustered according to mixed method.
    """

    model_name = [
        'sym_Hc_space-MNISymC3_K-32',
        'sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-32',
        'sym_MdPoNiIbWmDeSo_space-MNISymC3_K-32',
        'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32',
        'sym_MdPoNiIbWmDeSo_space-MNISymC3_K-32_meth-mixed',
        'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed_fromC3']

    # Evaluate
    results = evaluation(model_name, test_datasets)

    # Save file
    fname = 'eval_' + model_name.split(' / ')[-1] + '.tsv'
    results.to_csv(res_dir + fname, index=False, sep='\t')


if __name__ == "__main__":
    # evaluate_clustered()
    evaluate_sym(K=[68], train_type=[
                 'all', 'indiv'], rest_included=True, out_file='eval_sym_68_rest_all.tsv')
    evaluate_sym(K=[68], train_type=[
        'all', 'loo', 'indiv'], rest_included=False, out_file='eval_sym_68_task_all.tsv')
    # evaluate_sym(K=[68], train_type=['loo',
    #              'all'], rest_included=True, out_file='eval_sym_68_rest_loo_all.tsv')
    pass
