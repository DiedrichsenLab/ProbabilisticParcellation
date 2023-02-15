#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for learning fusion on datasets

Created on 02/15/2023 at 2:16 PM
Author: cnettekoven
"""
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.learn_fusion_gpu as lfg
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


def fit_models(ks, fit_datasets=['all', 'loo', 'indiv'], rest_included=False, verbose=True):

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
        dataset_list.extend(individual_datasets)

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    for datasets in dataset_list:
        for k in ks:
            datanames = ''.join(T.two_letter_code[datasets])
            wdir = ut.model_dir + f'/Models/Models_{t}'
            fname = f'/sym_{datanames}_space-{space}_K-{k}.tsv'

            if not Path(wdir + fname).exists():
                print(
                    f'fitting model {t} with K={k} in space {space} as {fname}...')
                if verbose:
                    ut.report_cuda_memory()
                lfg.fit_all(datasets, k, model_type=t, repeats=100,
                            sym_type=[msym], space=space)
            else:
                print(
                    f'model {t} with K={k} in space {space} already fitted as {fname}')


if __name__ == "__main__":
    fit_models(ks=[32], fit_datasets=['all'], rest_included=False)
    fit_models(ks=[32], fit_datasets=['all'], rest_included=True)
