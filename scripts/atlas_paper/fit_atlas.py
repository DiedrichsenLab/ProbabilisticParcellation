#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for learning fusion on datasets

Created on 02/15/2023 at 2:16 PM
Author: cnettekoven
"""
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.learn_fusion_gpu as lf
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import Functional_Fusion.matrix as matrix
import nibabel as nb
import generativeMRF.full_model as fm
import generativeMRF.spatial as sp
import generativeMRF.arrangements as ar
import generativeMRF.emissions as em
import generativeMRF.evaluation as ev
import torch as pt
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import time


def fit_asym_from_sym_sep_hem(mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68', mname_new=None):
    # Load model
    inf, m = ut.load_batch_best(mname)
    inf = ut.recover_info(inf, m, mname)
    # Freeze emission model and fit arrangement model
    M, new_info = lf.refit_model(m, inf, fit='arrangement', sym_new='asym')
    # save new model
    if mname_new is None:
        mname_new = f'{mname.split("/")[0]}asym_{mname.split("sym_")[1]}_arrange-asym'
    with open(f'{ut.model_dir}/Models/{mname_new}.pickle', 'wb') as file:
        pickle.dump([M], file)
        # save new info
        new_info.to_csv(f'{ut.model_dir}/Models/{mname_new}.tsv',
                        sep='\t', index=False)
        print(
            f'Done. Saved asymmetric model as: \n\t{mname_new} \nOutput folder: \n\t{ut.model_dir}/Models/ \n\n')
    return M, new_info


def fit_asym_from_sym(mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'):
    # Load model
    inf, m = ut.load_batch_best(mname)
    inf = ut.recover_info(inf, m, mname)
    # Freeze emission model and fit arrangement model
    M, new_info = lf.refit_model(m, inf, fit='arrangement', sym_new='asym')
    # save new model
    mname_new = f'asym_{mname.split("sym_")[1]}_arrange-asym'
    with open(f'{ut.model_dir}/Models/{mname_new}.pickle', 'wb') as file:
        pickle.dump([M], file)
        # save new info
        new_info.to_csv(f'{ut.model_dir}/Models/{mname_new}.tsv',
                        sep='\t', index=False)
        print(
            f'Done. Saved asymmetric model as: \n\t{mname_new} \nOutput folder: \n\t{ut.model_dir}/Models/ \n\n')
    return M, new_info


def fit_models(ks, fit_datasets=['all', 'loo', 'indiv'], rest_included=False, verbose=True, indiv_on_rest_only=False):

    ########## Settings ##########
    space = 'MNISymC3'  # Set atlas space
    msym = 'sym'  # Set model symmetry
    t = '03'  # Set model type

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
    if 'all' in fit_datasets:
        dataset_list.extend([alldatasets])
    if 'loo' in fit_datasets:
        dataset_list.extend(loo_datasets)
    if 'indiv' in fit_datasets:
        if indiv_on_rest_only:
            dataset_list.append([7])
        else:
            dataset_list.extend(individual_datasets)

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    for datasets in dataset_list:
        for k in ks:
            datanames = ''.join(T.two_letter_code[datasets])
            wdir = ut.model_dir + f'/Models/Models_{t}'
            fname = f'/{msym}_{datanames}_space-{space}_K-{k}.tsv'

            if not Path(wdir + fname).exists():
                print(
                    f'fitting model {t} with K={k} in space {space} as {fname}...')
                if verbose:
                    ut.report_cuda_memory()
                lf.fit_all(datasets, k, model_type=t, repeats=100,
                           sym_type=[msym], space=space)
            else:
                print(
                    f'model {t} with K={k} in space {space} already fitted as {fname}')


if __name__ == "__main__":
    # ks = [10, 20, 34, 40, 68]
    # ks = [28, 30, 36, 38, 74]
    # ks = [68, 80]
    # fit_models(ks=[32], fit_datasets=['all'], rest_included=False)
    # fit_models(ks=ks, fit_datasets=['indiv', 'all'],
    #            rest_included=True, indiv_on_rest_only=True)
    # fit_models(ks=ks, fit_datasets=['loo'], rest_included=True)

    # fit_asym_from_sym(
    # mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68')
    fit_asym_from_sym_sep_hem(
        mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68', mname_new='Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem')
