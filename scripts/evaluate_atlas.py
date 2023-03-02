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
import generativeMRF.evaluation as gev

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
import ProbabilisticParcellation.functional_profiles as fp
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

            # 1. Run DCBC
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


def evaluate_models(ks, model_types=['all', 'loo', 'indiv'], model_on=['task', 'rest'], test_on='task'):

    model_datasets = get_model_datasets(model_on, model_types)
    ########## Settings ##########
    space = 'MNISymC3'  # Set atlas space
    msym = 'sym'  # Set model symmetry
    t = '03'  # Set model type

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    for datasets in model_datasets:
        for k in ks:
            datanames = ''.join(T.two_letter_code[datasets])
            wdir = ut.model_dir + f'/Models/Models_{t}'
            mname = f'Models_03/{msym}_{datanames}_space-{space}_K-{k}'
            fname = f'eval_on-{test_on}_' + mname.split('/')[-1] + '.tsv'

            if Path(res_dir + fname).exists():
                print(f'File {fname} already exists. Skipping.')
            else:
                print(
                    f'\nEvaluating {mname}...\nTrained on {T.name.iloc[datasets].tolist()}.')
                test_datasets = get_test_datasets(model_on, test_on, datasets)
                test_datasets = T.name.iloc[test_datasets].tolist()
                # Evaluate
                results = evaluation(mname, test_datasets)

                # Save file
                results.to_csv(res_dir + fname, index=False, sep='\t')


def get_model_datasets(model_on, model_types, indiv_on_rest_only=False):
    """Evalute models that were clustered according to mixed method.
    """
    # -- Build dataset list --

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    n_dsets = len([r for r, row in T.iterrows()
                  if row.behaviour_state in model_on])

    alldatasets = np.arange(n_dsets).tolist()
    loo_datasets = [np.delete(np.arange(n_dsets), d).tolist()
                    for d in alldatasets]
    individual_datasets = [[d] for d in alldatasets]

    dataset_list = []
    if 'all' in model_types:
        dataset_list.extend([alldatasets])
    if 'loo' in model_types:
        dataset_list.extend(loo_datasets)
    if 'indiv' in model_types:
        if indiv_on_rest_only:
            dataset_list.append([7])
        else:
            dataset_list.extend(individual_datasets)

    return dataset_list


def get_test_datasets(model_on, test_on, model_datasets):
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    eligible_datasets = T[T.behaviour_state.isin(model_on)].index.tolist()
    test_datasets = T[T.behaviour_state == test_on].index.tolist()

    # Make sure you test on left out dataset only for leave-one-out models
    left_out_dataset = list(set(eligible_datasets) - set(model_datasets))
    if len(left_out_dataset) == 1:
        # If difference between modelled datasets and all eligible datasets is 1, then model must be leave-one-out
        print(
            f'Detected Leave-One-Out Model. Testing on left out dataset: {T.name.iloc[left_out_dataset[0]]}')
        test_datasets = left_out_dataset

    return test_datasets


def evaluate_selected(on='task'):
    """Evalute selected models on task data.
    """

    model_name = [
        'Models_03/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-68',
        'Models_03/sym_Hc_space-MNISymC3_K-80'
        'Models_03/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-80']

    if on == 'task':
        test_datasets = ['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                         'WMFS', 'Demand', 'Somatotopic']
    elif on == 'rest':
        test_datasets = ['HCP']

    for m, mname in enumerate(model_name):
        fname = f'eval_on-{on}_' + mname.split('/')[-1] + '.tsv'

        if Path(res_dir + fname).exists():
            print(f'File {fname} already exists. Skipping.')
        else:
            # Evaluate
            results = evaluation(mname, test_datasets)

            # Save file
            results.to_csv(res_dir + fname, index=False, sep='\t')


def evaluate_existing(test_on='task'):
    """Evalute existing parcellations (MDTB, Buckner) on task data.
    """
    parcels = ['Anatom', 'MDTB10', 'Buckner7', 'Buckner17', 'Ji10']

    test_datasets = [0, 1, 2, 3, 4, 5, 6, 7]

    par_name = []
    for p in parcels:
        par_name.append(ut.base_dir + '/Atlases/tpl-MNI152NLin2009cSymC/' +
                        f'atl-{p}_space-MNI152NLin2009cSymC_dseg.nii')

    pass


def compare_models(ks, model_types=['all', 'loo', 'indiv'], model_on=['task', 'rest']):
    """
    Compare models based on their Adjusted Rand Index (ARI)

    Args:
    - ks: list of integers (parcel numbers for models)
    - model_types: list of strings representing the model types to be used for comparison. Default: ['all', 'loo', 'indiv']
    - model_on: list of strings representing the data type (task, rest or both) on which the models were fitted. Default: ['task', 'rest']


    """

    model_datasets = get_model_datasets(model_on, model_types)

    ########## Settings ##########
    space = 'MNISymC3'  # Set atlas space
    msym = 'sym'  # Set model symmetry
    t = '03'  # Set model type

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')

    results = pd.DataFrame()

    for k in ks:
        model_names = [
            f'Models_03/{msym}_{" ".join(T.two_letter_code[datasets])}_space-{space}_K-{k}' for datasets in model_datasets]
        loaded_models, loaded_info = get_models(
            model_names)
        combinations = [(model_names[i], model_names[j]) for i in range(len(model_names))
                        for j in range(i + 1, len(model_names))]

        r = compare_ari(loaded_models, loaded_info, combinations)
        results = pd.concat([results, r], ignore_index=True)

    model_type = '-'.join(model_types)
    ks = [str(k) for k in ks]
    ks = '-'.join(ks)
    results.to_csv(
        res_dir + f'ARI_{msym}_{model_type}_space-{space}_K-{ks}_.tsv', index=False, sep='\t')


def get_models(model_names):
    """
    Load models for comparison

    Args:
    - model_names: list of strings representing the names of the models to be loaded

    Returns:
    - loaded_models: dictionary with keys as the model names and values as the loaded models
    - loaded_info: dictionary with keys as the model names and values as the model information
    """

    info_all = []
    models_all = []
    for i in model_names:
        info, model = ut.load_batch_best(i)
        info_all.append(info)
        models_all.append(model)

    loaded_models = dict(zip(model_names, models_all))
    loaded_info = dict(zip(model_names, info_all))

    return loaded_models, loaded_info


def compare_ari(combinations, loaded_models, loaded_info):
    """
    Compute Adjusted Rand Index (ARI) between pairs of loaded models

    Args:
    - combinations: list of tuples representing the combinations of models to be compared
    - loaded_models: dictionary with keys as the model names and values as the loaded models
    - loaded_info: dictionary with keys as the model names and values as the model information

    Returns:
    - results: DataFrame with columns 'model_name', 'atlas', 'K', 'train_data', 'train_loglik', and 'ari'
    """

    results = pd.DataFrame()
    for (a, b) in combinations:
        # load models
        info_a = loaded_info[a]
        model_a = loaded_models[a]
        info_b = loaded_info[b]
        model_b = loaded_models[b]

        ari_group = gev.ARI(pt.argmax(model_a.arrange.marginal_prob(), dim=0), pt.argmax(
            model_b.arrange.marginal_prob(), dim=0))

        # 1. Run ARI
        res_ari = pd.DataFrame({'model_name': [info_a['name'], info_b['name']],
                                'atlas': info_a.atlas,
                                'K': info_a.K,
                                'train_data': [info_a.datasets, info_b.datasets],
                                'train_loglik': [info_a.loglik, info_b.loglik],
                                'ari': ari_group,
                                })
        results = pd.concat([results, res_ari], ignore_index=True)

    return results


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
    # evaluate_models(ks, model_types=['loo'], model_on=[
    #                 'task'], test_on='task')
    # evaluate_models(ks, model_types=['loo'], model_on=[
    #                 'task', 'rest'], test_on='task')

    # evaluate_existing(on='task')

    compare_models(ks=ks, model_types=['indiv'], model_on=['task', 'rest'])
    pass
