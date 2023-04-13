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
from cortico_cereb_connectivity import evaluation as cev
from datetime import datetime
import seaborn as sns
import re

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

    parcels = ['Anatom', 'MDTB10', 'Buckner7', 'Ji10', 'Buckner17']
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
    if combinations == 'all':
        combinations = list(it.combinations(loaded_models.keys(), 2))

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


def calc_ari(parcels):
    """
    Calculate ARI between all parcellations in a list of parcellations

    Args:
    - parcels (np.array or list): parcellations to be compared

    Returns:
    - ari (np.array): ARI matrix

    """
    if isinstance(parcels[0], list):  # if parcels are two lists of parcellations to compare
        ari = np.zeros((len(parcels[0]), len(parcels[1])))
        for i in range(len(parcels[0])):
            for j in range(len(parcels[1])):
                ari[i, j] = gev.ARI(parcels[0][i], parcels[1][j]).item()
    else:
        ari = np.zeros((len(parcels), len(parcels)))
        for i in range(len(parcels)):
            for j in range(len(parcels)):
                ari[i, j] = gev.ARI(parcels[i], parcels[j]).item()
    return ari


def load_existing_parcellations(space):
    """
    Load existing parcellations

    Args:
    - space (str): atlas space

    Returns:
    - existing (list): list of existing parcellations
    - labels (list): list of labels for existing parcellations

    """

    # Load existing parcellations
    existing = ['/tpl-MNI152NLin2009cSymC/atl-Buckner7_space-MNI152NLin2009cSymC_dseg.nii',
                '/tpl-MNI152NLin2009cSymC/atl-Ji10_space-MNI152NLin2009cSymC_dseg.nii',
                '/tpl-MNI152NLin2009cSymC/atl-Buckner17_space-MNI152NLin2009cSymC_dseg.nii',
                '/tpl-MNI152NLin2009cSymC/atl-MDTB10_space-MNI152NLin2009cSymC_dseg.nii',
                '/tpl-MNI152NLin2009cSymC/atl-Anatom_space-MNI152NLin2009cSymC_dseg.nii', ]
    labels = ['Buckner7', 'Ji10', 'Buckner17', 'MDTB10', 'Anatom']
    atlas, _ = am.get_atlas(
        space, atlas_dir=ut.base_dir + '/Atlases')
    parcels = []
    for i in range(len(existing)):
        par = nb.load(ut.atlas_dir + existing[i])
        Pgroup = pt.tensor(atlas.read_data(par, 0) + 1,
                           dtype=pt.get_default_dtype())
        parcels.append(Pgroup)

    return parcels, labels


def load_individual_parcellations(ks, space):
    """Returns parcellations for individual datasets

    Args:
    - ks (list): list of integers representing the number of parcels for each model
    - space (str): atlas space

    Returns:
    - parcels (list): list of parcellations
    - labels (list): list of labels for parcellations
    """
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')

    model_datasets = [[0], [1], [2], [3], [4], [5], [6], [7]]

    # Load individual parcellations
    # for idx, k in enumerate(ks):
    for idx, datasets in enumerate(model_datasets):
        model_names = [
            f'Models_03/asym_{"".join(T.two_letter_code[datasets])}_space-{space}_K-{k}' for k in ks]
        loaded_models, _ = get_models(
            model_names)
        # get model parcellations
        parcels = [(pt.argmax(model.arrange.marginal_prob(), dim=0))
                   for model in loaded_models.values()]

        # Make labels
        labels = [
            f'{"".join(T.two_letter_code[datasets])}_asym_{k}' for k in ks]

        if idx == 0:
            all_parcels = [parcels]
            all_labels = [labels]
        else:
            all_parcels.append(parcels)
            all_labels.append(labels)

    return all_parcels, all_labels


def get_compMat(criterion='ari', ks=[10, 20, 34, 68], model_types=['all', 'loo', 'indiv'], sym=['sym', 'asym'], space='MNISymC3'):
    """
    Gets the comparison matrix for the given criterion

    Args:
    - criterion: string representing the criterion to be used for MDS. Default: 'ARI'
    - ks: list of integers (parcel numbers for models)
    - model_types: list of strings representing the model types to be used for comparison. Default: ['all', 'loo', 'indiv']
    - model_on: list of strings representing the data type (task, rest or both) on which the models were fitted. Default: ['task', 'rest']
    - compare: string representing the comparison to be made. Default: 'train_data'

    Returns:
    - compMat: comparison matrix
    - labels: list of labels for the comparison matrix

    """
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')

    # Load models
    for idx, k in enumerate(ks):
        model_datasets = [[0], [1], [2], [3], [4], [5], [6], [
            0, 1, 2, 3, 4, 5, 6], [7], [0, 1, 2, 3, 4, 5, 6, 7]]
        model_names = [
            f'Models_03/{msym}_{"".join(T.two_letter_code[datasets])}_space-{space}_K-{k}' for datasets in model_datasets for msym in sym]
        loaded_models, _ = get_models(
            model_names)
        # get model parcellations
        parcels = [(pt.argmax(model.arrange.marginal_prob(), dim=0))
                   for model in loaded_models.values()]

        # Make labels
        labels = [
            f'{"".join(T.two_letter_code[datasets])}_{msym}' for datasets in model_datasets for msym in sym]
        informative_labels = {'MdPoNiIbWmDeSo_sym': 'Task_sym',
                              'MdPoNiIbWmDeSoHc_sym': 'Task+Rest_sym',
                              'Hc_sym': 'Rest_sym',
                              'MdPoNiIbWmDeSo_asym': 'Task_asym',
                              'MdPoNiIbWmDeSoHc_asym': 'Task+Rest_asym',
                              'Hc_asym': 'Rest_asym'}
        labels = [informative_labels[label] if label in informative_labels.keys(
        ) else label for label in labels]

        # load existing
        existing, existing_labels = load_existing_parcellations(space)
        parcels = parcels + existing
        labels = labels + existing_labels

        # Calculate ARI
        results = calc_ari(parcels)

        # Store results in third dimension
        if idx == 0:
            compMat = np.zeros(
                (len(parcels), len(parcels), len(ks)))
        compMat[:, :, idx] = results
        idx += 1

    return compMat, labels


def plot_mds(Results, labels):

    # Average across Ks
    Results_avg = np.mean(Results, axis=2)
    # ---------------- Plotting ----------------

    # ---- Plot correlation matrix ----
   # Create a mask to hide the diagonal for visualisation
    mask = np.zeros_like(Results_avg)
    mask[np.diag_indices(Results_avg.shape[0])] = True

    # TODO: Derive eigenvectors from only task data and then project the data

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(Results_avg, mask=mask, annot=True, ax=ax, cmap='RdYlBu_r')
    ax.set_xticklabels(
        labels)
    ax.set_yticklabels(
        labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # ---- Plot MDS ----
    # Remove Anatomical parcellation before calculating MDS
    labels_func = labels[:-1]
    results_func = Results_avg[:-1, :-1]

    # Calculate the eigenvalues and eigenvectors of the correlation coefficient matrix
    eigenvalues, eigenvectors = np.linalg.eig(results_func)

    # Choose the two eigenvectors with the highest eigenvalues
    idx = eigenvalues.argsort()[::-1][:2]
    pos = eigenvectors[:, idx]

    # Project the data onto the eigenvectors of just the functional data?

    # Plot the resulting points
    plt.figure()
    plt.scatter(pos[:, 0], pos[:, 1])
    for j in range(pos.shape[0]):
        plt.text(pos[j, 0] + 0.005, pos[j, 1], labels[j],
                 fontdict=dict(alpha=0.5))

    # ---- Plot 3D MDS ----
    # Choose the three eigenvectors with the highest eigenvalues
    idx = eigenvalues.argsort()[::-1][:3]
    pos = eigenvectors[:, idx]
    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    for j in range(pos.shape[0]):
        ax.text(pos[j, 0] + 0.005, pos[j, 1], pos[j, 2], labels[j],
                fontdict=dict(alpha=0.5))

    pass


def compare_across_granularity(ks, verbose=False, exsting_included=True):
    """Calculate the ARI between every granularity of parcellation 1 and every granularity of parcellation 2

    """
    space = 'MNISymC3'
    # Get models at different granularities
    parcels, labels = load_individual_parcellations(ks, space)
    # Get existing Rest parcellations at different granularities (Buckner7, Ji10, Buckner17)
    if exsting_included:
        existing, existing_labels = load_existing_parcellations(space)
        existing_gran = [existing[0], existing[1], existing[2]]
        labels_gran = [existing_labels[0],
                       existing_labels[1], existing_labels[2]]
        parcels.append(existing_gran)
        labels.append(labels_gran)
    # TODO: Get existing MDTB parcellations at different granularities (MDTB07, MDTB10, MDTB17)

    # Build subplot with all matrices
    ARI = np.zeros((len(parcels), len(parcels)))
    aris = []
    a = 0
    for i, parcel1 in enumerate(parcels):
        for j, parcel2 in enumerate(parcels):
            ari = calc_ari([parcel1, parcel2])
            if verbose:
                print(f'ARI between {labels[i]} and {labels[j]}: {ari}')

            # Store labels for matrix
            if a == 0:
                labels_vs = []
            labels_vs.append(f'{labels[i]}_{labels[j]}')

            # Store mean ARI in matrix
            ARI[i, j] = np.mean(ari)
            ARI[j, i] = np.mean(ari)
            # Store entire matrix
            aris.append(ari)
            a += 1
    return ARI, aris, labels, parcels


def plot_comp_matrix(aris, labels, parcels):

    # Get numbers at the end of the labels
    granularity_labels = []
    dataset_labels = []
    for label in labels:
        granularity_labels.append([re.findall(r'\d+', l)[0]
                                   for l in label])
        dataset_labels.append([re.findall(r'\D+', l)[0]
                               for idx, l in enumerate(label) if idx == 0][0])

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))
    grid = (len(parcels), len(parcels))
    a = 0
    for i, parcel1 in enumerate(parcels):
        for j, parcel2 in enumerate(parcels):

            plt.subplot(grid[0], grid[1], i * grid[0] + j + 1)
            sns.heatmap(aris[a], annot=False, vmin=0, vmax=0.8)
            # sns.heatmap(aris[a], annot=False)
            # plot axis labels
            if i == 0:  # first row
                plt.title(dataset_labels[j])
            if j == 0:  # first column
                plt.ylabel(dataset_labels[i])

            # Remove xticks and yticks for each matrix
            plt.xticks([])
            plt.yticks([])

            a += 1


def average_comp_matrix(aris):
    n_parcellations = int(np.sqrt(len(aris)))
    # Average off-diagonal elements of each matrix
    ARI_avg = np.zeros((n_parcellations, n_parcellations))
    for i in np.arange(n_parcellations):
        for j in np.arange(n_parcellations):
            ari = aris[i * n_parcellations + j]
            if i == j:
                mask = np.zeros_like(ari, dtype=bool)
                mask[np.diag_indices(ari.shape[0])] = True
                ARI_avg[i, j] = np.mean(ari[mask == False])
            else:
                ARI_avg[i, j] = np.mean(ari)
    return ARI_avg


def norm_comp_matrix(aris, ARI_avg):
    n_parcellations = int(np.sqrt(len(aris)))

    # Average off-diagonal elements of each matrix
    ARI_norm = np.zeros((n_parcellations, n_parcellations))
    aris_norm = []
    for i in np.arange(n_parcellations):
        for j in np.arange(n_parcellations):
            ari = aris[i * n_parcellations + j]

            if i == j:
                mask = np.zeros_like(ari, dtype=bool)
                mask[np.diag_indices(ari.shape[0])] = True
                ari_norm = ari[mask == False] / \
                    np.sqrt(ARI_avg[i, i] * ARI_avg[j, j])
            else:
                ari_norm = ari / np.sqrt(ARI_avg[i, i] * ARI_avg[j, j])

            ARI_norm[i, j] = np.mean(ari_norm)
            aris_norm.append(ari_norm)

    # Normalize each row
    return ARI_norm, aris_norm


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

    # mname1 = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'
    # mname2 = 'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_reordered'
    # comp = compare_voxelwise(mname1,
    #                          mname2, plot=True, method='ari', save_nifti=True)
    # comp = compare_voxelwise(mname1,
    #                          mname2, plot=True, method='ri', save_nifti=True)
    # comp = ev.compare_voxelwise(mname1,
    #                             mname2, plot=True, method='corr', save_nifti=False, lim=(0, 1))
    # comp = ev.compare_voxelwise(mname1,
    #                             mname2, plot=True, method='cosang', save_nifti=False)

    # compMat, labels = get_compMat(criterion='ari', ks=[10, 20, 34, 40, 68], model_types=[
    #     'all', 'indiv'], sym=['asym'])
    # plot_mds(compMat, labels)

    # compMat, labels = get_compMat(criterion='ari', ks=[10, 20, 34, 40, 68], model_types=[
    #     'all', 'indiv'], sym=['asym'])

    ARI, aris, labels, parcels = compare_across_granularity(
        ks=[10, 20, 34, 40, 68], exsting_included=False)

    # Normalize aris by within-dataset reliability
    ARI_avg = average_comp_matrix(aris)
    ARI_norm, aris_norm = norm_comp_matrix(aris, ARI_avg)

    granularity_labels = []
    dataset_labels = []
    for label in labels:
        granularity_labels.append([re.findall(r'\d+', l)[0]
                                   for l in label])
        dataset_labels.append([re.findall(r'\D+', l)[0]
                               for idx, l in enumerate(label) if idx == 0][0])

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(ARI_avg, annot=True, vmin=0, vmax=0.5, ax=ax,
                xticklabels=dataset_labels, yticklabels=dataset_labels)
    plt.title('Average ARI between granularities')
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(ARI_norm, annot=True, vmin=0, vmax=0.5, ax=ax,
                xticklabels=dataset_labels, yticklabels=dataset_labels)
    plt.title('Average ARI between granularities')
    plt.show()

    # Test whether task based datasets are more similar to MDTB than to HCP
    n_parcellations = int(np.sqrt(len(aris)))
    mdtb_row = dataset_labels.index('Md_asym_')
    hcp_row = dataset_labels.index('Hc_asym_')

    mdtb_values = [aris_norm[i * n_parcellations + j]
                   for j in np.arange(n_parcellations) for i in np.arange(n_parcellations) if i == mdtb_row and j != mdtb_row and j < hcp_row]
    mdtb_values = [el for arr in mdtb_values for row in arr for el in row]

    hcp_values = [aris_norm[i * n_parcellations + j]
                  for j in np.arange(hcp_row) for i in np.arange(n_parcellations) if i == hcp_row and j < hcp_row]
    hcp_values = [el for arr in hcp_values for row in arr for el in row]

    task_values = [aris_norm[i * n_parcellations + j]
                   for j in np.arange(hcp_row) for i in np.arange(j + 1, hcp_row) if i != j]
    task_values = [el for arr in task_values for row in arr for el in row]

    import scipy.stats as stats
    print(stats.ttest_ind(mdtb_values, hcp_values))
    print(np.mean(mdtb_values), np.mean(hcp_values))
    print(stats.ttest_ind(task_values, hcp_values))
    print(np.mean(task_values), np.mean(hcp_values))

    pass
