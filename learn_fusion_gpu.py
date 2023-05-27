#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for learning fusion on datasets

Created on 11/17/2022 at 2:16 PM
Author: dzhi, jdiedrichsen
"""
import ProbabilisticParcellation.util as ut
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import Functional_Fusion.matrix as matrix
import nibabel as nb
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.evaluation as ev
import torch as pt
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import time
# TEMPORARY FIX BEFORE MODELS ARE UPDATED TO HBP - REMOVE!!! TODO XX
import generativeMRF.arrangements as gar


def build_data_list(datasets,
                    atlas='MNISymC3',
                    sess=None,
                    cond_ind=None,
                    type=None,
                    part_ind=None,
                    subj=None,
                    join_sess=True,
                    join_sess_part=False):
    """Builds list of datasets, cond_vec, part_vec, subj_ind
    from different data sets
    Args:
        datasets (list): Names of datasets to include
        atlas (str): Atlas indicator
        sess (list): list of 'all' or list of sessions
        design_ind (list, optional): _description_. Defaults to None.
        part_ind (list, optional): _description_. Defaults to None.
        subj (list, optional): _description_. Defaults to None.
        join_sess (bool, optional): Model the sessions with a single model . Defaults to True.
    Returns:
        data,
        cond_vec,
        part_vec,
        subj_ind
    """
    n_sets = len(datasets)
    data = []
    cond_vec = []
    part_vec = []
    subj_ind = []

    # Set defaults for data sets:
    if sess is None:
        sess = ['all'] * n_sets
    if part_ind is None:
        part_ind = [None] * n_sets
    if cond_ind is None:
        cond_ind = [None] * n_sets
    if type is None:
        type = [None] * n_sets

    sub = 0
    # Run over datasets get data + design
    for i in range(n_sets):
        dat, info, ds = get_dataset(ut.base_dir, datasets[i],
                                    atlas=atlas,
                                    sess=sess[i],
                                    type=type[i])
        # Sub-index the subjects:
        if subj is not None:
            dat = dat[subj[i], :, :]
        n_subj = dat.shape[0]

        # Find correct indices
        if cond_ind[i] is None:
            cond_ind[i] = ds.cond_ind
        if part_ind[i] is None:
            part_ind[i] = ds.part_ind
        # Make different sessions either the same or different
        if join_sess:
            data.append(dat)
            cond_vec.append(info[cond_ind[i]].values.reshape(-1, ))

            # Check if we want to set no partition after join sessions
            if join_sess_part:
                part_vec.append(np.ones(info[part_ind[i]].shape))
            else:
                part_vec.append(info[part_ind[i]].values.reshape(-1, ))
            subj_ind.append(np.arange(sub, sub + n_subj))
        else:
            if sess[i] == 'all':
                sessions = ds.sessions
            else:
                sessions = sess[i]
            # Now build and split across the correct sessions:
            for s in sessions:
                indx = info.sess == s
                data.append(dat[:, indx, :])
                cond_vec.append(info[cond_ind[i]].values[indx].reshape(-1, ))
                part_vec.append(info[part_ind[i]].values[indx].reshape(-1, ))
                subj_ind.append(np.arange(sub, sub + n_subj))
        sub += n_subj
    return data, cond_vec, part_vec, subj_ind


def build_model(K, arrange, sym_type, emission, atlas,
                cond_vec, part_vec,
                uniform_kappa=True,
                weighting=None):
    """ Builds a Full model based on your specification"""
    if arrange == 'independent':
        if sym_type == 'sym':
            ar_model = ar.ArrangeIndependentSymmetric(K,
                                                      atlas.indx_full,
                                                      atlas.indx_reduced,
                                                      same_parcels=False,
                                                      spatial_specific=True,
                                                      remove_redundancy=False)
        elif sym_type == 'asym':
            ar_model = ar.ArrangeIndependent(K, atlas.P,
                                             spatial_specific=True,
                                             remove_redundancy=False)
    else:
        raise (NameError(f'unknown arrangement model:{arrange}'))

    # Initialize emission models
    em_models = []
    for j, ds in enumerate(cond_vec):
        if emission == 'VMF':
            em_model = em.MixVMF(K=K, P=atlas.P,
                                 X=matrix.indicator(cond_vec[j]),
                                 part_vec=part_vec[j],
                                 uniform_kappa=uniform_kappa)
        elif emission == 'GMM':
            em_model = em.MixGaussian(K=K, P=atlas.P,
                                      X=matrix.indicator(cond_vec[j]),
                                      std_V=False)
        elif emission == 'wVMF':
            em_model = em.wMixVMF(K=K, P=atlas.P,
                                  X=matrix.indicator(cond_vec[j]),
                                  part_vec=part_vec[j],
                                  uniform_kappa=uniform_kappa,
                                  weighting='lsquare_sum2P')
        else:
            raise ((NameError(f'unknown emission model:{emission}')))
        em_models.append(em_model)
    M = fm.FullMultiModel(ar_model, em_models)
    if weighting is not None:
        M.ds_weight = weighting  # Weighting for each dataset

    return M


def batch_fit(datasets, sess,
              type=None, cond_ind=None, part_ind=None, subj=None,
              atlas=None,
              K=10,
              arrange='independent',
              sym_type='asym',
              emission='VMF',
              n_rep=3, n_inits=10, n_iter=80, first_iter=10,
              name=None,
              uniform_kappa=True,
              join_sess=True,
              join_sess_part=False,
              weighting=None):
    """ Executes a set of fits starting from random starting values
    selects the best one from a batch and saves them

    Args:
        datasets (list): List of dataset names to be used as training
        sess (list): List of list of sessions to be used for each
        type (list): List the data types
        cond_ind (list): Name of the info-field that indicates the condition
        part_ind (list): Name of the field indicating independent partitions of the data
        subj (list, optional): _description_. Defaults to None
        atlas (Atlas): Atlas to be used. Defaults to None.
        K (int): Number of parcels. Defaults to 10.
        arrange (str): Type of arangement model. Defaults to 'independent'.
        sym_type (str): {'sym','asym'} - defaults to asymmetric model
        emission (list / strs): Type of emission models. Defaults to 'VMF'.
        n_inits (int): Number of random starting values. default: 10
        n_iter (int): Maximal number of iterations per fit: default: 20
        save (bool): Save the resulting fits? Defaults to True.
        name (str): Name of model (for filename). Defaults to None.

    Returns:
        info (pd.DataFrame):
    """
    print(f'Start loading data: {datasets} - {sess} - {type} ...')
    tic = time.perf_counter()
    data, cond_vec, part_vec, subj_ind = build_data_list(datasets,
                                                         atlas=atlas.name,
                                                         sess=sess,
                                                         cond_ind=cond_ind,
                                                         type=type,
                                                         part_ind=part_ind,
                                                         subj=subj,
                                                         join_sess=join_sess,
                                                         join_sess_part=join_sess_part)
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    # Load all necessary data and designs
    n_sets = len(data)

    print(f'Building fullMultiModel {arrange} + {emission} for fitting...')
    M = build_model(K, arrange, sym_type, emission, atlas,
                    cond_vec, part_vec,
                    uniform_kappa, weighting)
    fm.report_cuda_memory()

    # Initialize data frame for results
    models, priors = [], []
    n_fits = n_rep
    info = pd.DataFrame({'name': [name] * n_fits,
                         'atlas': [atlas.name] * n_fits,
                         'K': [K] * n_fits,
                         'datasets': [datasets] * n_fits,
                         'sess': [sess] * n_fits,
                         'type': [type] * n_fits,
                         'subj': [subj] * n_fits,
                         'arrange': [arrange] * n_fits,
                         'emission': [emission] * n_fits,
                         'loglik': [np.nan] * n_fits,
                         'weighting': [weighting] * n_fits})

    # Iterate over the number of fits
    ll = np.empty((n_fits, n_iter))
    prior = pt.zeros((n_fits, K, atlas.P))
    for i in range(n_fits):
        print(f'Start fit: repetition {i} - {name}')

        iter_tic = time.perf_counter()
        # Copy the object (without data)
        m = deepcopy(M)
        # Attach the data
        m.initialize(data, subj_ind=subj_ind)
        fm.report_cuda_memory()

        m, ll, theta, U_hat, ll_init = m.fit_em_ninits(
            iter=n_iter,
            tol=0.01,
            fit_arrangement=True,
            n_inits=n_inits,
            first_iter=first_iter, verbose=False)
        info.loglik.at[i] = ll[-1].cpu().numpy()  # Convert to numpy
        m.clear()

        # Align group priors
        if i == 0:
            indx = pt.arange(K)
        else:
            indx = ev.matching_greedy(prior[0, :, :], m.marginal_prob())
        prior[i, :, :] = m.marginal_prob()[indx, :]

        this_similarity = []
        for j in range(i):
            # Option1: K*K similarity matrix between two Us
            # this_crit = cal_corr(prior[i, :, :], prior[j, :, :])
            # this_similarity.append(1 - pt.diagonal(this_crit).mean())

            # Option2: L1 norm between two Us
            this_crit = pt.abs(prior[i, :, :] - prior[j, :, :]).mean()
            this_similarity.append(this_crit)

        num_rep = sum(sim < 0.02 for sim in this_similarity)
        print(num_rep)

        # Move to CPU device before storing
        m.move_to(device='cpu')
        models.append(m)

        # Convergence: 1. must run enough repetitions (50);
        #              2. num_rep greater than threshold (10% of max_iter)
        if (i > 50) and (num_rep >= int(n_fits * 0.1)):
            break
        iter_toc = time.perf_counter()
        print(
            f'Done fit: repetition {i} - {name} - {iter_toc - iter_tic:0.4f} seconds!')

    models = np.array(models, dtype=object)

    return info, models


def fit_all(set_ind=[0, 1, 2, 3], K=10, repeats=100, model_type='01',
            sym_type=['asym', 'sym'], subj_list=None, weighting=None, this_sess=None, space=None):
    # Get dataset info
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    datasets = T.name.to_numpy()
    sess = np.array(['all'] * len(T), dtype=object)
    if this_sess is not None:
        for i, idx in enumerate(set_ind):
            sess[idx] = this_sess[i]

    type = T.default_type.to_numpy()
    cond_ind = T.default_cond_ind.to_numpy()
    part_ind = np.array(['half'] * len(T), dtype=object)

    # Make the atlas object
    if space is None:
        space = 'MNISymC3'

    atlas, _ = am.get_atlas(space, ut.atlas_dir)

    # Provide different setttings for the different model types
    join_sess_part = False
    if model_type == '01':
        uniform_kappa = True
        join_sess = True
    elif model_type == '02':
        uniform_kappa = False
        join_sess = True
    elif model_type[:6] == '01-HCP':
        uniform_kappa = True
        weighting = np.repeat(1, len(set_ind) - 1).tolist()
        hcp_weight = model_type.split('HCP')[1]
        weighting.extend([float(f'{hcp_weight[0]}.{hcp_weight[1]}')])
        join_sess = True
    elif model_type == '03':
        uniform_kappa = True
        join_sess = False
    elif model_type == '04':
        uniform_kappa = False
        join_sess = False
    elif model_type == '05':
        uniform_kappa = False
        join_sess = True
        join_sess_part = True
    elif model_type == '06':
        uniform_kappa = True
        join_sess = True
        join_sess_part = True

    # Generate a dataname from first two letters of each training data set
    dataname = ''.join(T.two_letter_code[set_ind])

    for mname in sym_type:
        tic = time.perf_counter()
        name = mname + '_' + ''.join(dataname)
        info, models = batch_fit(datasets[set_ind],
                                 sess=sess[set_ind],
                                 type=type[set_ind],
                                 cond_ind=cond_ind[set_ind],
                                 part_ind=part_ind[set_ind],
                                 subj=subj_list,
                                 atlas=atlas,
                                 K=K,
                                 sym_type=mname,
                                 name=name,
                                 n_inits=50,
                                 n_iter=200,
                                 n_rep=repeats,
                                 first_iter=30,
                                 join_sess=join_sess,
                                 join_sess_part=join_sess_part,
                                 uniform_kappa=uniform_kappa,
                                 weighting=weighting)

        # Save the fits and information
        wdir = ut.model_dir + f'/Models/Models_{model_type}'
        fname = f'/{name}_space-{atlas.name}_K-{K}'

        if this_sess is not None:
            return wdir, fname, info, models

        if subj_list is not None:
            wdir = ut.model_dir + f'/Models/Models_{model_type}/leaveNout'
            fname = f'/{name}_space-{atlas.name}_K-{K}'
            return wdir, fname, info, models

        info.to_csv(wdir + fname + '.tsv', sep='\t')
        with open(wdir + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)

        toc = time.perf_counter()
        print(f'Done Model fitting - {mname}. Used {toc - tic:0.4f} seconds!')


def clear_models(K, model_type='04'):
    for t in ['sym', 'asym']:
        for k in K:
            for s in ['MdPoNiIbHc_00', 'MdPoNiIbHc_02',
                      'MdPoNiIbHc_10']:  # Md','Po','Ni','Hc','Ib','MdPoNiIb','MdPoNiIbHc','MdPoNiIbHc_00']:
                fname = f"Models_{model_type}/{t}_{s}_space-MNISymC3_K-{k}"
                try:
                    ut.clear_batch(fname)
                    print(f"cleared {fname}")
                except:
                    print(f"skipping {fname}")


def leave_one_out_fit(dataset=[0], model_type=['01'], K=10):
    # Define some constant
    nsubj = [24, 8, 6, 12, 100]
    ########## Leave-one-out fitting ##########
    for m in model_type:
        this_nsub = nsubj[dataset[0]]
        for i in range(this_nsub):
            print(
                f'fitting dataset:{dataset} - model:{m} - leaveNout: {i} ...')
            sub_list = np.delete(np.arange(this_nsub), i)
            wdir, fname, info, models = fit_all(dataset, K,
                                                model_type=m,
                                                sym_type=['asym'],
                                                subj_list=[sub_list])
            fname = fname + f'_leave-{i}'
            info.to_csv(wdir + fname + '.tsv', sep='\t')
            with open(wdir + fname + '.pickle', 'wb') as file:
                pickle.dump(models, file)


def fit_indv_sess(indx=3, model_type='01', K=10):
    datasets = np.array(['MDTB', 'Pontine', 'Nishimoto',
                         'IBC', 'WMFS', 'Demand', 'Somatotopic'],
                        dtype=object)
    _, _, my_dataset = get_dataset(ut.base_dir, datasets[indx])
    sess = my_dataset.sessions
    for indv_sess in sess:
        ibc_dir = ut.model_dir + f'/Models/Models_{model_type}'
        nam = f'/asym_Ib_space-MNISymC3_K-{K}_{indv_sess}'

        if not Path(ibc_dir + nam + '.tsv').exists():
            print(
                f'fitting model {model_type} with K={K} on IBC sessions {indv_sess} ...')
            wdir, fname, info, models = fit_all([indx], K,
                                                model_type=model_type,
                                                repeats=100,
                                                sym_type=['asym'],
                                                this_sess=[[indv_sess]])
            fname = fname + f'_{indv_sess}'
            info.to_csv(wdir + fname + '.tsv', sep='\t')
            with open(wdir + fname + '.pickle', 'wb') as file:
                pickle.dump(models, file)


def fit_two_IBC_sessions(K=10, sess1='clips4', sess2='rsvplanguage', model_type='04'):
    ibc_dir = ut.model_dir + f'/Models/Models_{model_type}/IBC_sessFusion'
    nam = f'/asym_Ib_space-MNISymC3_K-{K}_ses-{sess1}+{sess2}'

    if not Path(ibc_dir + nam + '.tsv').exists():
        print(
            f'fitting model {model_type} with K={K} on IBC sessions {sess1} + {sess2} ...')
        wdir, fname, info, models = fit_all([3], K, model_type=model_type, repeats=50,
                                            sym_type=['asym'], this_sess=[['ses-' + sess1,
                                                                           'ses-' + sess2]])
        fname = fname + f'_ses-{sess1}+{sess2}'
        info.to_csv(wdir + '/IBC_sessFusion' + fname + '.tsv', sep='\t')
        with open(wdir + '/IBC_sessFusion' + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)


def fit_all_datasets(space='MNISymC2',
                     msym='sym',
                     K=[68],
                     datasets_list=[[0, 1, 2, 3, 4, 5, 6]]):
    # -- Model fitting --
    # datasets_list = [[0], [1], [2], [3], [4], [5], [6], [0, 1, 2, 3, 4, 5, 6, 7]]

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    # for i in range(7):
    #     datasets = [0, 1, 2, 3, 4, 5, 6]
    #     datasets.remove(i)
    for datasets in datasets_list:
        for k in K:
            for t in ['03', '04']:
                datanames = ''.join(T.two_letter_code[datasets])
                wdir = ut.model_dir + f'/Models'
                fname = f'/Models_{t}/{msym}_{datanames}_space-{space}_K-{k}'

                # inf, m = load_batch_fit(fname)
                # if not m[0].ds_weight.is_cuda:
                #     print(f'Convert model {t} with K={k} {fname} to cuda...')
                #     # move_batch_to_device(fname, device='cuda')
                if not Path(wdir + fname + '.tsv').exists():
                    print(f'fitting model {t} with K={k} as {fname}...')
                    fit_all(datasets, k, model_type=t,
                            repeats=100, sym_type=[msym])
                else:
                    print(f'model {t} with K={k} already fitted as {fname}')


def refit_model(model, new_info, fit='emission', sym_new=None):
    """Refits emission models.

    Args:
        model:      Model to be refitted
        new_info:       Information for new model

    Returns:
        model: Refitted model

    """

    if sym_new is None and (type(model.arrange) is ar.ArrangeIndependentSymmetric or type(model.arrange) is gar.ArrangeIndependentSeparateHem):
        M = fm.FullMultiModel(model.arrange, model.emissions)

    elif sym_new == 'asym' and type(model.arrange) is ar.ArrangeIndependentSymmetric:
        atlas, _ = am.get_atlas(new_info.atlas, ut.atlas_dir)
        indx_hem = np.sign(atlas.world[0, :])
        # Make indx_hem two-dimensional with the same entries
        # Add empty row dimension to index_hem to make it 2D
        indx_hem = indx_hem[np.newaxis, :]

        # Make arrangement model asymmetric but with hemispheres fitted separately
        new_arrange = ar.ArrangeIndependentSeparateHem(model.K,
                                                       indx_hem=indx_hem,
                                                       spatial_specific=model.arrange.spatial_specific,
                                                       remove_redundancy=model.arrange.rem_red,
                                                       )
        M = fm.FullMultiModel(new_arrange, model.emissions)
        M.nsubj = model.nsubj
        M.n_emission = model.n_emission
        M.nsubj_list = model.nsubj_list
        M.subj_ind = model.subj_ind
        if hasattr(model, 'ds_weight'):
            M.ds_weight = model.ds_weight

        # Update arrangement model parameters to get the symmetric log likelihoods into the asymmetric model
        new_logpi = model.arrange.map_to_full(model.arrange.logpi)
        # Halven the logpi for the reduced number of parcels
        new_logpi = new_logpi[new_logpi.shape[0] // 2:]
        M.arrange.logpi = new_logpi
        M.arrange.set_param_list(['logpi'])

    model_settings = {'Models_01': [True, True, False],
                      'Models_02': [False, True, False],
                      'Models_03': [True, False, False],
                      'Models_04': [False, False, False],
                      'Models_05': [False, True, True]}

    # uniform_kappa = model_settings[new_info.model_type][0]
    join_sess = model_settings[new_info.model_type][1]
    join_sess_part = model_settings[new_info.model_type][2]

    datasets = new_info.datasets
    sessions = new_info.sess
    types = new_info.type

    data, cond_vec, part_vec, subj_ind = build_data_list(datasets,
                                                         atlas=new_info.atlas,
                                                         sess=sessions,
                                                         type=types,
                                                         join_sess=join_sess,
                                                         join_sess_part=join_sess_part)
    # Attach the data
    M.initialize(data, subj_ind=subj_ind)

    if fit == 'emission':

        # Refit emission models
        print(f'Freezing arrangement model and fitting emission models...\n')

        M, ll, _, _ = M.fit_em(iter=500, tol=0.01,
                               fit_emission=True,
                               fit_arrangement=False,
                               first_evidence=True)

        # make info from a Series back to a dataframe
        if type(new_info) is not pd.DataFrame:
            new_info = pd.DataFrame(new_info.to_dict(), index=[0])
        new_info['loglik'] = ll[-1].item()

    elif fit == 'arrangement':
        # Refit arrangement model
        print(f'Freezing emission models and fitting arrangement model...\n')

        M, ll, _, _ = M.fit_em(iter=500, tol=0.01,
                               fit_emission=False,
                               fit_arrangement=True,
                               first_evidence=True)

        # make info from a Series back to a dataframe
        new_info = new_info.to_frame().T
        new_info['loglik'] = ll[-1].item()

    # Plot ll
    #
    # pt.Tensor.ndim = property(lambda self: len(self.shape))
    # x = pt.linspace(0,ll.shape[0], ll.shape[0])
    # plt.figure()
    # plt.plot(x[0:], ll[0:])

    return M, new_info


if __name__ == "__main__":
    datasets_list = [0]
    K = 17
    sym_type = ['asym']
    model_type = '03'
    space = 'MNISymC3'

    for k in [10, 20, 34, 40, 68, 100]:
        fit_all(set_ind=datasets_list, K=k, repeats=100, model_type=model_type,
                sym_type=['asym'], space='MNISymC3')

    pass
