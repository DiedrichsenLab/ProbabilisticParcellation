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


def evaluate_sym(K=[68], train_type=['indiv', 'loo', 'all'], space='MNISymC3', rest_included=False, out_file=None):
    """Evaluate models
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

    model_name = [f'Models_03/sym_{train_dset}_space-{space}_K-{this_k}'
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


def evaluate_models(ks, evaluate_datasets=['all', 'loo', 'indiv'], rest_included=False, verbose=True, indiv_on_rest_only=False, on='task'):

    ########## Settings ##########
    space = 'MNISymC3'  # Set atlas space
    msym = 'sym'  # Set model symmetry
    t = '03'  # Set model type

    if on == 'task':
        test_datasets = ['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                         'WMFS', 'Demand', 'Somatotopic']
    elif on == 'rest':
        test_datasets = ['HCP']

    # -- Build dataset list --
    if rest_included:
        n_dsets = 8  # with HCP
    else:
        n_dsets = 7  # without HCP
    alldatasets = np.arange(n_dsets).tolist()
    loo_datasets = [np.delete(np.arange(n_dsets), d).tolist()
                    for d in alldatasets]
    individual_datasets = [[d] for d in alldatasets]

    dataset_list = []
    if 'all' in evaluate_datasets:
        dataset_list.extend([alldatasets])
    if 'loo' in evaluate_datasets:
        dataset_list.extend(loo_datasets)
    if 'indiv' in evaluate_datasets:
        if indiv_on_rest_only:
            dataset_list.append([7])
        else:
            dataset_list.extend(individual_datasets)

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    for datasets in dataset_list:
        for k in ks:
            datanames = ''.join(T.two_letter_code[datasets])
            wdir = ut.model_dir + f'/Models/Models_{t}'
            mname = f'Models_03/{msym}_{datanames}_space-{space}_K-{k}'
            fname = f'eval_on-{on}_' + mname.split('/')[-1] + '.tsv'

            if Path(res_dir + fname).exists():
                print(f'File {fname} already exists. Skipping.')
            else:
                # Make sure you test on left out dataset only for leave-one-out models
                if (rest_included and len(datasets) == 6) or (rest_included and len(datasets) == 7):
                    left_out_dataset = list(set(alldatasets) - set(datasets))
                    test_datasets = [T.name[left_out_dataset].item()]

                # Evaluate
                results = evaluation(mname, test_datasets)

                # Save file
                results.to_csv(res_dir + fname, index=False, sep='\t')


def evaluate_clustered(test_datasets=['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                                      'WMFS', 'Demand', 'Somatotopic', 'HCP']):
    """Evalute models that were clustered according to mixed method.
    """

    model_name = [
        'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed']

    # Evaluate
    results = evaluation(model_name, test_datasets)

    # Save file
    fname = 'eval_' + model_name.split('/')[-1] + '.tsv'
    results.to_csv(res_dir + fname, index=False, sep='\t')


def evaluate_selected(on='task'):
    """Evalute selected models on task data.
    """

    if on == 'task':
        test_datasets = ['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                         'WMFS', 'Demand', 'Somatotopic']
    elif on == 'rest':
        test_datasets = ['HCP']

    model_name = [
        'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-80'
        'Models_03/sym_Hc_space-MNISymC3_K-80'
        'Models_03/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-80',

    ]

    for m, mname in enumerate(model_name):
        fname = f'eval_on-{on}_' + mname.split('/')[-1] + '.tsv'

        if Path(res_dir + fname).exists():
            print(f'File {fname} already exists. Skipping.')
        else:
            # Evaluate
            results = evaluation(mname, test_datasets)

            # Save file
            results.to_csv(res_dir + fname, index=False, sep='\t')


if __name__ == "__main__":
    # evaluate_clustered()
    # evaluate_sym(K=[68], train_type=[
    #              'all', 'indiv'], rest_included=True, out_file='eval_sym_68_rest_all.tsv')
    # evaluate_sym(K=[68], train_type=[
    #     'all', 'loo', 'indiv'], rest_included=False, out_file='eval_sym_68_task_all.tsv')
    # evaluate_sym(K=[68], train_type=['loo',
    #              'all'], rest_included=True, out_file='eval_sym_68_rest_loo_all.tsv')

    # evaluate_selected(on='task')
    # evaluate_selected(on='rest')

    ks = [10, 20, 34, 40, 68]
    evaluate_models(on='task', ks=ks, evaluate_datasets=[
        'loo'], rest_included=False)
    evaluate_models(on='task', ks=ks, evaluate_datasets=[
        'loo'], rest_included=True)
    pass
