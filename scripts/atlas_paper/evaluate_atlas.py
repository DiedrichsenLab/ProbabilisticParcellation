#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripts for atlas evaluation
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


def evaluation(model_name, test_datasets, tseries=False):
    """
    Evaluate a given model on a number of specified test datasets.

    Args:
        model_name (str): Name of the model to be evaluated.
        test_datasets (list of str): List of test datasets to be evaluated on.
        tseries (bool, optional): Whether the test data is in timeseries format. Defaults to False.

    Returns:
        pd.DataFrame: Results of the evaluation.
    """

    # Cross Valudation setting
    CV_setting = [('half', 1), ('half', 2)]

    # # determine space:
    space = model_name.split('space-')[-1].split('_')[0]

    # Get atlas
    atlas, _ = am.get_atlas(space, atlas_dir=ut.base_dir + '/Atlases')

    # Get results
    Results = pd.DataFrame()
    for dset in test_datasets:
        print(f'Testdata: {dset}\n')
        if tseries and dset == 'HCP':
            results = evaluate_timeseries(
                model_name, dset, atlas, CV_setting)
        else:
            results = evaluate_standard(model_name, dset, atlas, CV_setting)
        Results = pd.concat([Results, results])
    return Results


def evaluate_standard(model_name, dset, atlas, CV_setting):
    """
    Evaluate a given model on a standard dataset.

    Args:
        model_name (str): Name of the model to be evaluated.
        dset (str): Name of the dataset to be evaluated on.
        atlas (ndarray): Atlas used for the evaluation.
        CV_setting (list of tuples): Cross-validation settings.

    Returns:
        pd.DataFrame: Results of the evaluation.
    """
    # # determine space:
    space = model_name.split('space-')[-1].split('_')[0]

    tdata, tinfo, tds = ds.get_dataset(
        ut.base_dir, dset, atlas=space, sess='all')

    # default from dataset class
    cond_ind = tds.cond_ind
    cond_vec = tinfo[cond_ind].values.reshape(-1, )

    part_vec = tinfo['half'].values
    # part_vec = np.ones((tinfo.shape[0],), dtype=int)

    ################ CV starts here ################
    results = pd.DataFrame()
    for (indivtrain_ind, indivtrain_values) in CV_setting:
        # If type is tseries, then evaluate each subject separately - otherwise data is too large

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


def evaluate_timeseries(model_name, dset, atlas, CV_setting):
    # # determine space:
    space = model_name.split('space-')[-1].split('_')[0]

    cond_ind = 'time_id'
    tdata, tinfo, tds = ds.get_dataset(
        ut.base_dir, dset, atlas=space, sess='all', type='Tseries', info_only=True)

    cond_vec = tinfo[cond_ind].values.reshape(-1, )
    part_vec = tinfo['half'].values

    results = pd.DataFrame()
    for (indivtrain_ind, indivtrain_values) in CV_setting:
        # This can only be run on the Heavy server, not the GPU server
        train_indx = tinfo[indivtrain_ind] == indivtrain_values
        test_indx = tinfo[indivtrain_ind] != indivtrain_values
        res_dcbc = ev.run_dcbc(model_name, tdata, atlas,
                               train_indx=train_indx,
                               test_indx=test_indx,
                               cond_vec=cond_vec,
                               part_vec=part_vec,
                               device=ut.default_device,
                               verbose=False)
        res_dcbc['test_data'] = dset + '-Tseries'
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
                if test_on == 'tseries':
                    tseries = True
                    test_datasets = get_test_datasets(
                        model_on, test_on='rest', model_datasets=datasets)
                else:
                    test_datasets = get_test_datasets(
                        model_on, test_on=test_on, model_datasets=datasets)
                test_datasets = T.name.iloc[test_datasets].tolist()

                # Evaluate
                results = evaluation(mname, test_datasets, tseries=tseries)

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


def evaluate_selected(test_on='task'):
    """Evalute selected models on task data.
    """

    model_name = [
        'Models_03/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-68',
        'Models_03/sym_Hc_space-MNISymC3_K-80'
        'Models_03/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-80']

    if test_on == 'task':
        test_datasets = ['MDTB', 'Pontine', 'Nishimoto', 'IBC',
                         'WMFS', 'Demand', 'Somatotopic']
    elif test_on == 'rest':
        test_datasets = ['HCP']

    for m, mname in enumerate(model_name):
        fname = f'eval_on-{test_on}_' + mname.split('/')[-1] + '.tsv'

        if Path(res_dir + fname).exists():
            print(f'File {fname} already exists. Skipping.')
        else:
            # Evaluate
            results = evaluation(mname, test_datasets)

            # Save file
            results.to_csv(res_dir + fname, index=False, sep='\t')


def evaluate_existing(test_on='task', models=None):
    """Evalute existing parcellations (MDTB, Buckner).
    """

    parcels = ['Anatom', 'MDTB10', 'Buckner7', 'Buckner17', 'Ji10']
    space = 'MNISymC3'
    if models is None:
        models = ['Models_03/asym_Md_space-MNISymC3_K-10.pickle']

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    if test_on == 'task':
        test_datasets = [0, 1, 2, 3, 4, 5, 6]
        test_datasets = T.name.iloc[test_datasets].tolist()
    elif test_on == 'rest':
        test_datasets = [7]
        test_datasets = T.name.iloc[test_datasets].tolist()
    elif test_on == ['task', 'rest']:
        test_on = 'task+rest'
        test_datasets = [0, 1, 2, 3, 4, 5, 6, 7]
        test_datasets = T.name.iloc[test_datasets].tolist()
    elif test_on == ['tseries']:
        test_on = 'tseries'
        test_datasets = [7]
        test_datasets = T.name.iloc[test_datasets].tolist()

    par_name = []
    for p in parcels:
        par_name.append(ut.base_dir + '/Atlases/tpl-MNI152NLin2009cSymC/' +
                        f'atl-{p}_space-MNI152NLin2009cSymC_dseg.nii')
    par_name = models + par_name

    fname = f'eval_on-{test_on}_existing.tsv'

    if Path(res_dir + fname).exists():
        print(f'File {fname} already exists. Skipping.')
    else:
        print(
            f'\nEvaluating existing parcellations...\nTest on {test_on}.')
        results = pd.DataFrame()
        for dset in test_datasets:
            print(f'Testdata: {dset}\n')
            if test_on == 'tseries' and dset == 'HCP':

                tdata, tinfo, tds = ds.get_dataset(
                    ut.base_dir, dset, atlas=space, sess='all', type='Tseries', info_only=True)
                res_dcbc = pd.DataFrame()
                # This can only be run on the Heavy server, not the GPU server
                res_sub_sess = ev.run_dcbc_group(par_name,
                                                 space=space,
                                                 test_data=dset + '-Tseries',
                                                 test_sess='all',
                                                 tdata=tdata,
                                                 verbose=True)
                res_dcbc = pd.concat(
                    [res_dcbc, res_sub_sess], ignore_index=True)
            else:
                res_dcbc = ev.run_dcbc_group(par_name,
                                             space=space,
                                             test_data=dset,
                                             test_sess='all')
            # Concatenate results
            results = pd.concat([results, res_dcbc], ignore_index=True)
        results.to_csv(res_dir + fname, index=False, sep='\t')

    pass


def ARI_voxelwise(U_1, U_2):
    """Compute the adjusted rand index between two parcellations for all voxels.
    Args:
        U_1: First parcellation (usually estimated Us from fitted model 1)
        U_2: Second parcellation (usually estimated Us from fitted model 2)
    Returns:
        Vector containing the adjusted rand index for all voxels
    """
    # Initialize matrices
    sameReg_U_1 = (U_1[:, None] == U_1).int()
    sameReg_U_2 = (U_2[:, None] == U_2).int()
    sameReg_U_1.fill_diagonal_(0)
    sameReg_U_2.fill_diagonal_(0)

    # Compute ARI for each voxel
    # Initialize vector
    ARI_voxelwise = pt.zeros(U_1.shape[0], dtype=pt.float32)
    for i in range(U_1.shape[0]):
        # Compute ARI for voxel i and all other voxels
        sameReg_U_1_voxel = sameReg_U_1[:, i]
        sameReg_U_2_voxel = sameReg_U_2[:, i]

        # Get voxel pairs that are in the same parcel in both U_1 and U_2
        n_11_voxel = (sameReg_U_1_voxel * sameReg_U_2_voxel).sum()

        # Get voxel pairs that are in different parcels in both U_1 and U_2
        n_00_voxel = (1 - sameReg_U_1_voxel) * (1 - sameReg_U_2_voxel)
        # Set indices where voxel is compared to itself to 0
        n_00_voxel[i] = 0
        n_00_voxel = n_00_voxel.sum()

        # Get voxel pairs that are in the same parcel in U_1 but different parcels in U_2
        tmp = sameReg_U_1_voxel - sameReg_U_2_voxel
        tmp[tmp < 0] = 0
        n_10_voxel = tmp.sum()

        # Get voxel pairs that are in the same parcel in U_2 but different parcels in U_1
        tmp = sameReg_U_2_voxel - sameReg_U_1_voxel
        tmp[tmp < 0] = 0
        n_01_voxel = tmp.sum()

        # Special cases: empty data or full agreement (tn, fp), (fn, tp)
        if pt.all(n_01_voxel == 0) and pt.all(n_10_voxel == 0):
            ari_voxel = pt.tensor(1.0)
        else:
            ari_voxel = 2.0 * (n_11_voxel * n_00_voxel - n_10_voxel * n_01_voxel) / ((n_11_voxel + n_10_voxel)
                                                                                     * (n_10_voxel + n_00_voxel) +
                                                                                     (n_11_voxel +
                                                                                      n_01_voxel))
        ARI_voxelwise[i] = ari_voxel

    return ARI_voxelwise


def compare_models_voxelwise(mname_A, mname_B, save_nifti=False):
    # load models
    info_a, model_a = ut.load_batch_best(mname_A)
    info_b, model_b = ut.load_batch_best(mname_B)
    atlas = info_a.atlas

    parcel_a = pt.argmax(model_a.arrange.marginal_prob(), dim=0)
    parcel_b = pt.argmax(model_b.arrange.marginal_prob(), dim=0)
    ari_voxelwise = ARI_voxelwise(parcel_a, parcel_b)
    ari = ari_voxelwise.numpy()

    # Plot ARIs on flatmap
    # Get 95th percentile of ARI_voxelwise for cscale limits
    ari_95 = np.percentile(ari, 95)
    ari_05 = np.percentile(ari, 5)

    ari_nan = ari.copy()
    ari_nan[ari < 0] = np.nan
    # Plot ARI_voxelwise on flatmap
    plt.figure()
    ut.plot_data_flat(ari_nan, atlas,
                      dtype='func',
                      render='matplotlib',
                      cscale=[ari_05, ari_95],
                      colorbar=True)
    plt.show()

    # Save ARI_voxelwise as nifti
    if save_nifti:
        suit_atlas, _ = am.get_atlas(atlas, ut.base_dir + '/Atlases')
        ari_data = suit_atlas.data_to_nifti(ari)

        fname = ut.base_dir + '/Results/' + mname_A + '_' + mname_B + '_ARI.nii'
        nb.save(ari_data, fname + f'_dseg.nii')

        print(f'Saved ARI image {fname}.')

    # Match

    # Compute how many voxels land in the same parcellation
    match = (parcel_a == parcel_b).int()

    # Plot match on flatmap
    plt.figure()
    ut.plot_data_flat(match, atlas,
                      dtype='label',
                      render='matplotlib',
                      colorbar=True)
    plt.show()

    pass


def compare_models(ks, model_types=['all', 'loo', 'indiv'], model_on=['task', 'rest'], compare='train_data'):
    """
    Compare models trained on different datasets on their Adjusted Rand Index (ARI)

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

    # get info for file name
    model_type = '-'.join(model_types)

    for k in ks:
        if compare == 'train_data':
            model_names = [
                f'Models_03/{msym}_{"".join(T.two_letter_code[datasets])}_space-{space}_K-{k}' for datasets in model_datasets]
            fname = res_dir + \
                f'ARI_{msym}_{model_type}_space-{space}_K-{k}_.tsv'
            combinations = [(model_names[i], model_names[j]) for i in range(len(model_names))
                            for j in range(i + 1, len(model_names))]

        elif compare == 'symmetry':
            combinations = [
                (f'Models_03/sym_{"".join(T.two_letter_code[datasets])}_space-{space}_K-{k}', f'Models_03/asym_{"".join(T.two_letter_code[datasets])}_space-{space}_K-{k}') for datasets in model_datasets]
            model_names = [m for c in combinations for m in c]
            fname = res_dir + \
                f'ARI_sym-asym_{model_type}_space-{space}_K-{k}_.tsv'

        loaded_models, loaded_info = get_models(
            model_names)

        results = compare_ari(combinations, loaded_models, loaded_info)
        results.to_csv(fname, index=False, sep='\t')


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

        print(f'ARI {a} vs {b}: {ari_group.item():.3f}')

        # 1. Run ARI
        res_ari = pd.DataFrame({'model_name_a': info_a['name'],
                                'model_name_b': info_b['name'],
                                'atlas': info_a.atlas,
                                'K': info_a.K,
                                'train_data_a': info_a.datasets,
                                'train_data_b': info_b.datasets,
                                'train_loglik_a': info_a.loglik,
                                'train_loglik_b': info_b.loglik,
                                'ari': ari_group.item()},
                               index=[0]
                               )
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

    # evaluate_selected(test_on='task')
    # evaluate_selected(test_on='rest')

    # ks = [10, 20, 34, 40, 68]
    # evaluate_models(ks, model_types=['loo'], model_on=[
    #                 'task'], test_on='task')
    # evaluate_models(ks, model_types=['all'], model_on=[
    # 'task'], test_on='tseries')

    # evaluate_existing(test_on='task')

    # compare_models(ks=ks, model_types=['indiv', 'all'], model_on=[
    #                'task', 'rest'], compare='train_data')

    # compare_models(ks=ks, model_types=['indiv', 'all'], model_on=[
    #                'task', 'rest'], compare='symmetry')

    # evaluate_existing(test_on=['task', 'rest'])

    # evaluate_existing(test_on=['tseries'])

    # compare_existing()

    mname1 = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'
    mname2 = 'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_reordered'
    ari = compare_models_voxelwise(mname1, mname2)
    pass
