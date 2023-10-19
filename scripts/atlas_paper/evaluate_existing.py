#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripts for atlas evaluation
"""
import ProbabilisticParcellation.evaluate as ev
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.functional_profiles as fp
import Functional_Fusion.dataset as dset
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from scipy import stats
import glob
import numpy as np
import os
import time
import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am
import torch as pt
import json
import nibabel as nb


# == To generate the eval_all_5existing_on_taskDatasets.tsv file, run the following functions:

def run_dcbc_existing(model_names, tdata, space, device=None, load_best=True, verbose=True):
    """ Calculates DCBC using a test_data set. The test data splitted into
        individual training and test set given by `train_indx` and `test_indx`.
        First we use individual training data to derive an individual
        parcellations (using the model) and evaluate it on test data.
        By calling function `calc_test_dcbc`, the Means of the parcels are
        always estimated on N-1 subjects and evaluated on the Nth left-out
        subject.
    Args:
        model_names (list or str): Name of model fit (tsv/pickle file)
        tdata (pt.Tensor or np.ndarray): test data set
        atlas (atlas_map): The atlas map object for calculating voxel distance
        train_indx (ndarray of index or boolean mask): index of individual
            training data
        test_indx (ndarray or index boolean mask): index of individual test
            data
        cond_vec (1d array): the condition vector in test-data info
        part_vec (1d array): partition vector in test-data info
        device (str): the device name to load trained model
        load_best (str): I don't know
    Returns:
        data-frame with model evalution of both group and individual DCBC
    """
    # Calculate distance metric given by input atlas
    atlas, ainf = am.get_atlas(space, atlas_dir=ut.base_dir + '/Atlases')
    dist = ut.compute_dist(atlas.world.T, resolution=1)
    # convert tdata to tensor
    if type(tdata) is np.ndarray:
        tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

    if not isinstance(model_names, list):
        model_names = [model_names]

    # Load atlas description json
    with open(ut.atlas_dir + '/atlas_description.json', 'r') as f:
        T = json.load(f)

    space_dir = T[space]['dir']
    space_name = T[space]['space']
    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    # Now loop over possible models we want to evaluate
    cw, cb = [], []
    for i, model_name in enumerate(model_names):
        print(f"Doing model {model_name}\n")
        if verbose:
            ut.report_cuda_memory()
        # load existing parcellation
        par = nb.load(ut.atlas_dir +
                      f'/{space_dir}/atl-{model_name}_space-{space_name}_dseg.nii')
        Pgroup = pt.tensor(atlas.read_data(par, 0),
                           dtype=pt.get_default_dtype())
        Pgroup = pt.where(Pgroup==0, pt.tensor(float('nan')), Pgroup)
        this_res = pd.DataFrame()
        # ------------------------------------------
        # Now run the DCBC evaluation fo the group only
        dcbc_group = ev.calc_test_dcbc(Pgroup, tdata, dist,max_dist=110)

        # ------------------------------------------
        # Collect the information from the evaluation
        # in a data frame
        ev_df = pd.DataFrame({'model_name': [model_name] * num_subj,
                              'atlas': [space] * num_subj,
                              'K': [Pgroup.unique().shape[0]-1] * num_subj,
                              'train_data': [model_name] * num_subj,
                              'subj_num': np.arange(num_subj)})
        # Add all the evaluations to the data frame
        ev_df['dcbc_group'] = dcbc_group.cpu()
        ev_df['dcbc_indiv'] = np.nan
        results = pd.concat([results, ev_df], ignore_index=True)

    return results

def eval_existing(model_name, t_datasets=['MDTB','Pontine','Nishimoto'],
                  type=None, out_name=None, save=True, plot_wb=True):
    """Evaluate group and individual DCBC and write in evaluation file
    Args:
        K: the number of parcels
    Returns:
        Write in evaluation file
    """
    if not isinstance(model_name, list):
        model_name = [model_name]

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    corrW, corrB = [], []
    for i, dset in enumerate(t_datasets):
        print(f'Testdata: {dset}\n')
        # Preparing atlas, cond_vec, part_vec
        tic = time.perf_counter()
        tdata, tinfo, tds = ds.get_dataset(ut.base_dir, dset, atlas='MNISymC3',
                                        sess='all', type=type[i])
        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')

        if type[i] == 'Tseries':
            tds.cond_ind = 'time_id'

        res_dcbc, corr_w, corr_b = run_dcbc_existing(model_name, tdata, 'MNISymC3',
                                                     device='cuda')

        corrW.append(corr_w)
        corrB.append(corr_b)
        res_dcbc['test_data'] = dset
        results = pd.concat([results, res_dcbc], ignore_index=True)

    return results, corrW, corrB
    

# == To generate the eval_dataset7_sym.tsv and eval_dataset7_asym-hem.tsv files, loop through Ks (10, 20, 34, 40, 68) and datasets run the following functions:
#   -- Evaluate symmetric --
# model_name = ['Md','Po','Ni','Ib','Wm','De','So','MdPoNiIbWmDeSo', 'MdNiIbWmDeSo', 'MdPoIbWmDeSo', 'MdPoNiWmDeSo', 'MdPoNiIbDeSo', 'MdPoNiIbWmSo', 'MdPoNiIbWmDe']
# ks = [10, 20, 34, 40, 68]
# for K in ks:
#   result_5_eval(K=K, symmetric='sym', model_type=['03','04'], model_name=model_name)

#  -- Evaluate asymmetric fitted from symmetric --
# mname_suffix = '_arrange-asym_sep-hem'
# evaluate_models(ks, model_types=['indiv', 'loo', 'all'], model_on=[
# 'task'], test_on='task', mname_suffix=mname_suffix)

if __name__ == "__main__":
    # eval_existing(model_name=['Models_03/NettekovenSym32_space-MNISymC3'])
    test_datasets_list = [0,1,2,3,4,5,6,7]
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    num_subj = T.return_nsubj.to_numpy()[test_datasets_list]
    types = T.default_type.to_numpy()[test_datasets_list]
    results, _, _ = eval_existing(['Anatom', 'Buckner7', 'Buckner17', 'Ji10', 'MDTB10'],
              t_datasets=T.name.to_numpy()[test_datasets_list],
              type=types)
    

    # Save file
    wdir = ut.model_dir + f'/Models/Evaluation'
    fname = f'/eval_atlas_exsiting.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')