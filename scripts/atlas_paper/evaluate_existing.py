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

def run_dcbc_existing(atlas_names, tdata, space, max_dist=40, verbose=True):
    """ Calculates group DCBC using a test_data set. 

    Args:
        atlas_names (list or str): Name of existing atlases (dseg.nii files) to be evaluated.
        tdata (pt.Tensor or np.ndarray): Test data set to evaluate with the atlases.
        space (atlas_map): The atlas map object for calculating voxel distance.
        verbose (bool): Whether to print verbose messages during the evaluation (optional, default is True).

    Returns:
        pd.DataFrame: A DataFrame containing the model evaluation results of group DCBC.

    """
    # Calculate distance metric given by input atlas
    atlas, ainf = am.get_atlas(space, atlas_dir=ut.base_dir + '/Atlases')
    dist = ut.compute_dist(atlas.world.T, resolution=1)
    # convert tdata to tensor
    if type(tdata) is np.ndarray:
        tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

    if not isinstance(atlas_names, list):
        atlas_names = [atlas_names]

    # Load atlas description json
    with open(ut.atlas_dir + '/atlas_description.json', 'r') as f:
        T = json.load(f)

    space_dir = T[space]['dir']
    space_name = T[space]['space']
    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    # Now loop over possible atlasses we want to evaluate
    for i, atlas_name in enumerate(atlas_names):
        print(f"Doing model {atlas_name}\n")
        if verbose:
            ut.report_cuda_memory()
        # load existing parcellation
        par = nb.load(ut.atlas_dir +
                      f'/{space_dir}/atl-{atlas_name}_space-{space_name}_dseg.nii')
        Pgroup = pt.tensor(atlas.read_data(par, 0),
                           dtype=pt.get_default_dtype())
        Pgroup = pt.where(Pgroup==0, pt.tensor(float('nan')), Pgroup)
        this_res = pd.DataFrame()
        # ------------------------------------------
        # Run the DCBC evaluation for the group
        dcbc_group = ev.calc_test_dcbc(Pgroup, tdata, dist,max_dist=max_dist)

        # ------------------------------------------
        # Collect the information from the evaluation
        # in a data frame
        ev_df = pd.DataFrame({'atlas_name': [atlas_name] * num_subj,
                              'space': [space] * num_subj,
                              'K': [Pgroup.unique().shape[0]-1] * num_subj,
                              'subj_num': np.arange(num_subj)})
        # Add all the evaluations to the data frame
        ev_df['dcbc_group'] = dcbc_group.cpu()
        results = pd.concat([results, ev_df], ignore_index=True)

    return results

def eval_atlas(atlas_name, t_datasets=['MDTB','Pontine','Nishimoto'],
                  type=None, max_dist=40):
    """Evaluate group and individual DCBC and write in evaluation file
    Args:
        K: the number of parcels
    Returns:
        Write in evaluation file
    """
    if not isinstance(atlas_name, list):
        atlas_name = [atlas_name]

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    results = pd.DataFrame()
    # Evaluate atlasses on all test datasets
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

        res_dcbc = run_dcbc_existing(atlas_name, tdata, 'MNISymC3',
                                                    max_dist=max_dist)

        res_dcbc['test_data'] = dset
        results = pd.concat([results, res_dcbc], ignore_index=True)

    return results
    

if __name__ == "__main__":
    # eval_atlas(atlas_name=['Models_03/NettekovenSym32_space-MNISymC3'])
    test_datasets_list = [0,1,2,3,4,5,6,7]
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    num_subj = T.return_nsubj.to_numpy()[test_datasets_list]
    types = T.default_type.to_numpy()[test_datasets_list]
    existing_atlasses = ['Anatom', 'Buckner7', 'Buckner17', 'Ji10', 'MDTB10']
    fusion_atlasses = ['NettekovenAsym32', 'NettekovenSym32', 'NettekovenAsym68', 'NettekovenSym68']
    # max_dist = 110
    max_dist=70
    results = eval_atlas(fusion_atlasses + existing_atlasses,
              t_datasets=T.name.to_numpy()[test_datasets_list],
              type=types, max_dist=max_dist)
    

    # Save file
    wdir = ut.model_dir + f'/Models/Evaluation'
    fname = f'/eval_atlas_existing_dist-{max_dist}.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')






# == To generate the eval_dataset7_sym.tsv and eval_dataset7_asym-hem.tsv files, loop through Ks (10, 20, 34, 40, 68) and datasets run the following functions:
#   -- Evaluate symmetric --
# atlas_name = ['Md','Po','Ni','Ib','Wm','De','So','MdPoNiIbWmDeSo', 'MdNiIbWmDeSo', 'MdPoIbWmDeSo', 'MdPoNiWmDeSo', 'MdPoNiIbDeSo', 'MdPoNiIbWmSo', 'MdPoNiIbWmDe']
# ks = [10, 20, 34, 40, 68]
# for K in ks:
#   result_5_eval(K=K, symmetric='sym', model_type=['03','04'], atlas_name=atlas_name)

#  -- Evaluate asymmetric fitted from symmetric --
# mname_suffix = '_arrange-asym_sep-hem'
# evaluate_models(ks, model_types=['indiv', 'loo', 'all'], model_on=[
# 'task'], test_on='task', mname_suffix=mname_suffix)
