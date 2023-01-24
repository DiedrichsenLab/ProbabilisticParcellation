#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for learning fusion on datasets

Created on 11/17/2022 at 2:16 PM
Author: dzhi, jdiedrichsen
"""
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
from ProbabilisticParcellation.util import *
import torch as pt
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import time



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
        dat, info, ds = get_dataset(base_dir, datasets[i],
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

def build_model(K,arrange,sym_type,emission,atlas,
                cond_vec,part_vec,
                uniform_kappa = True,
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
              sym_type = 'asym',
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
    M = build_model(K,arrange,sym_type,emission,atlas,
                cond_vec,part_vec,
                uniform_kappa,weighting)
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
                         'weighting': [weighting] * n_fits});

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
            first_iter=first_iter)
        info.loglik.at[i] = ll[-1].cpu().numpy() # Convert to numpy
        m.clear()

        # Align group priors
        if i == 0:
            indx = pt.arange(K)
        else:
            indx = ev.matching_greedy(prior[0,:,:], m.marginal_prob())
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
        if (i>50) and (num_rep >= int(n_fits*0.1)):
            break
        iter_toc = time.perf_counter()
        print(f'Done fit: repetition {i} - {name} - {iter_toc - iter_tic:0.4f} seconds!')

    models = np.array(models, dtype=object)

    return info, models


def fit_all(set_ind=[0, 1, 2, 3], K=10, repeats=100, model_type='01',
            sym_type=['asym','sym'], subj_list=None, weighting=None, this_sess=None, space=None):
    # Get dataset info
    T = pd.read_csv(base_dir + '/dataset_description.tsv',sep='\t')
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
        space='MNISymC3'

    atlas, _ = am.get_atlas(space, atlas_dir)

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
                                 sym_type = mname,
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
        wdir = model_dir + f'/Models/Models_{model_type}'
        fname = f'/{name}_space-{atlas.name}_K-{K}'

        if this_sess is not None:
            return wdir, fname, info, models

        if subj_list is not None:
            wdir = model_dir + f'/Models/Models_{model_type}/leaveNout'
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
                    clear_batch(fname)
                    print(f"cleared {fname}")
                except:
                    print(f"skipping {fname}")


def write_dlabel_cifti(data, atlas,
                       labels=None,
                       label_names=None,
                       column_names=None,
                       label_RGBA=None):
    """Generates a label Cifti2Image from a numpy array

    Args:
        data (np.array):
            num_vert x num_col data
        atlas (obejct):
            the cortical surface <atlasSurface> object
        labels (list): Numerical values in data indicating the labels -
            defaults to np.unique(data)
        label_names (list):
            List of strings for names for labels
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors for labels
    Returns:
        gifti (GiftiImage): Label gifti image
    """
    if type(data) is pt.Tensor:
        if pt.cuda.is_available() or pt.backends.mps.is_built():
            data = data.cpu().numpy()
        else:
            data = data.numpy()

    if data.ndim == 1:
        # reshape to (1, num_vertices)
        data = data.reshape(-1, 1)

    num_verts, num_cols = data.shape
    if labels is None:
        labels = np.unique(data)
    num_labels = len(labels)

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i + 1))

    # Determine color scale if empty
    if label_RGBA is None:
        label_RGBA = [(0.0, 0.0, 0.0, 0.0)]
        if 0 in labels:
            num_labels -= 1
        hsv = plt.cm.get_cmap('hsv', num_labels)
        color = hsv(np.linspace(0, 1, num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        for i in range(num_labels):
            label_RGBA.append((color[i][0],
                               color[i][1],
                               color[i][2],
                               color[i][3]))

    # Create label names from numerical values
    if label_names is None:
        label_names = ['???']
        for i in labels:
            if i == 0:
                pass
            else:
                label_names.append("label-{:02d}".format(i))

    assert len(label_RGBA) == len(label_names), \
        "The number of parcel labels must match the length of colors!"
    labelDict = []
    for i, nam in enumerate(label_names):
        labelDict.append((nam, label_RGBA[i]))

    labelAxis = nb.cifti2.LabelAxis(column_names, dict(enumerate(labelDict)))
    header = nb.Cifti2Header.from_axes((labelAxis, atlas.get_brain_model_axis()))
    img = nb.Cifti2Image(dataobj=data.reshape(1, -1), header=header)

    return img

def save_cortex_cifti(fname):
    info, model = load_batch_best(fname)
    Prop = model.marginal_prob()
    par = pt.argmax(Prop, dim=0) + 1
    img = write_dlabel_cifti(par, am.get_atlas('fs32k', atlas_dir))
    nb.save(img, model_dir + f'/Models/{fname}.dlabel.nii')


def leave_one_out_fit(dataset=[0], model_type=['01'], K=10):
    # Define some constant
    nsubj = [24, 8, 6, 12, 100]
    ########## Leave-one-out fitting ##########
    for m in model_type:
        this_nsub = nsubj[dataset[0]]
        for i in range(this_nsub):
            print(f'fitting dataset:{dataset} - model:{m} - leaveNout: {i} ...')
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
    datasets = np.array(['MDTB','Pontine','Nishimoto',
                         'IBC','WMFS','Demand','Somatotopic'],
                        dtype=object)
    _, _, my_dataset = get_dataset(base_dir, datasets[indx])
    sess = my_dataset.sessions
    for indv_sess in sess:
        ibc_dir = model_dir + f'/Models/Models_{model_type}'
        nam = f'/asym_Ib_space-MNISymC3_K-{K}_{indv_sess}'

        if not Path(ibc_dir + nam + '.tsv').exists():
            print(f'fitting model {model_type} with K={K} on IBC sessions {indv_sess} ...')
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
    ibc_dir = model_dir + f'/Models/Models_{model_type}/IBC_sessFusion'
    nam = f'/asym_Ib_space-MNISymC3_K-{K}_ses-{sess1}+{sess2}'

    if not Path(ibc_dir + nam + '.tsv').exists():
        print(f'fitting model {model_type} with K={K} on IBC sessions {sess1} + {sess2} ...')
        wdir, fname, info, models = fit_all([3], K, model_type=model_type, repeats=50,
                                            sym_type=['asym'], this_sess=[['ses-'+sess1,
                                                                      'ses-'+sess2]])
        fname = fname + f'_ses-{sess1}+{sess2}'
        info.to_csv(wdir + '/IBC_sessFusion' + fname + '.tsv', sep='\t')
        with open(wdir + '/IBC_sessFusion' + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)

def fit_all_datasets(space = 'MNISymC2',
                    msym = 'sym',
                    K=[68],
                    datasets_list = [[0, 1, 2, 3, 4, 5, 6]]):
    # -- Model fitting --
    # datasets_list = [[0], [1], [2], [3], [4], [5], [6], [0, 1, 2, 3, 4, 5, 6, 7]]
    
    T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    # for i in range(7):
    #     datasets = [0, 1, 2, 3, 4, 5, 6]
    #     datasets.remove(i)
    for datasets in datasets_list:
        for k in K:
            for t in ['03', '04']:
                datanames = ''.join(T.two_letter_code[datasets])
                wdir = model_dir + f'/Models'
                fname = f'/Models_{t}/{msym}_{datanames}_space-{space}_K-{k}'

                # inf, m = load_batch_fit(fname)
                # if not m[0].ds_weight.is_cuda:
                #     print(f'Convert model {t} with K={k} {fname} to cuda...')
                #     # move_batch_to_device(fname, device='cuda')
                if not Path(wdir + fname + '.tsv').exists():
                    print(f'fitting model {t} with K={k} as {fname}...')
                    fit_all(datasets, k, model_type=t, repeats=100, sym_type=[msym])
                else:
                    print(f'model {t} with K={k} already fitted as {fname}')


def refit_model(model, new_info):
    """Refits model.

    Args:
        model:      Model to be refitted
        new_info:       Information for new model

    Returns:
        model: Refitted model

    """

    if type(model.arrange) is ar.ArrangeIndependentSymmetric:
        M = fm.FullMultiModel(model.arrange, model.emissions)
    else:
        M = fm.FullMultiModel(model.arrange, model.emissions)

    model_settings = {'Models_01': [True, True, False],
                      'Models_02': [False, True, False],
                      'Models_03': [True, False, False],
                      'Models_04': [False, False, False],
                      'Models_05': [False, True, True]}

    # uniform_kappa = model_settings[new_info.model_type][0]
    join_sess = model_settings[new_info.model_type][1]
    join_sess_part = model_settings[new_info.model_type][2]

    datasets = new_info.datasets.strip("'[").strip("]'").split("' '")
    sessions = new_info.sess.strip("'[").strip("]'").split("' '")
    types = new_info.type.strip("'[").strip("]'").split("' '")

    data, cond_vec, part_vec, subj_ind = build_data_list(datasets,
                                                         atlas=new_info.atlas,
                                                         sess=sessions,
                                                         type=types,
                                                         join_sess=join_sess,
                                                         join_sess_part=join_sess_part)

    # Attach the data
    M.initialize(data, subj_ind=subj_ind)

    # Refit emission models
    print(f'Freezing arrangement model and fitting emission models...\n')

    M, ll, _, _ = M.fit_em(iter=500, tol=0.01,
                                            fit_emission=True,
                                            fit_arrangement=False,
                                            first_evidence=True)
    
    # make info from a Series back to a dataframe
    new_info = pd.DataFrame(new_info.to_dict(), index=[0])
    new_info['loglik'] = ll[-1].item()
    # Plot ll
    # 
    # pt.Tensor.ndim = property(lambda self: len(self.shape))
    # x = pt.linspace(0,ll.shape[0], ll.shape[0])
    # plt.figure()
    # plt.plot(x[1:], ll[1:])


    return M, new_info


if __name__ == "__main__":
    datasets_list=[0,1,2,3,4,5,6]
    K = 68
    sym_type = ['asym']
    model_type = '03'
    space = 'MNISymC2'

    fit_all(set_ind=datasets_list, K=K, repeats=100, model_type=model_type,
            sym_type=['sym'], space='MNISymC2')


    ########## Reliability map
    # rel, sess = reliability_maps(base_dir, 'IBC', subtract_mean=False,
    #                              voxel_wise=True)
    # plt.figure(figsize=(25, 18))
    # plot_multi_flat(rel, 'MNISymC3', grid=(3, 5), dtype='func',
    #                 cscale=[-0.3, 0.7], colorbar=False, titles=sess)

    ########## IBC selected sessions fusion fit ##########
    sess_1 = DataSetIBC(base_dir + '/IBC').sessions
    sess_2 = DataSetIBC(base_dir + '/IBC').sessions
    for s1 in sess_1:
        sess_2.remove(s1)
        for s2 in sess_2:
            this_s1 = s1.split('-')[1]
            this_s2 = s2.split('-')[1]
            for k in [40, 68]:
                for t in ['01']:
                    wdir = model_dir + f'/Models/Models_{t}/IBC_sessFusion'
                    fname = wdir+f'/asym_Ib_space-MNISymC3_K-{k}_ses-{this_s1}+{this_s2}.tsv'
                    if not os.path.isfile(fname):
                        fit_two_IBC_sessions(K=k, sess1=this_s1, sess2=this_s2, model_type=t)
                        print(f'-Done type {t}, K={k}, IBC session {s1} and {s2} fusion.')

    ########## IBC all sessions fit ##########
    # fit_indv_sess(3, model_type='03', K=40)
    # fit_indv_sess(3, model_type='04', K=40)
    # dataset_list = [[0], [1], [2], [3], [0,1,2,3]]

    ########## IBC all fit ##########
    # fit_all([3], 34, model_type='04', repeats=100, sym_type=[0])


    ########## Leave-one-oout ##########
    # leave_one_out_fit(dataset=dataset_list, model_type=type_list, K=10)

    ########## Plot the flatmap results ##########
    # # Read the MDTB colors
    # color_file = atlas_dir + '/tpl-SUIT/atl-MDTB10.lut'
    # color_info = pd.read_csv(color_file, sep=' ', header=None)
    # MDTBcolors = color_info.iloc[:, 1:4].to_numpy()
    #
    # # Make IBC session model file names
    # # fnames = []
    # # sess = DataSetIBC(base_dir + '/IBC').sessions
    # # for s in sess:
    # #     fnames.append(f'Models_05/asym_Ib_space-MNISymC3_K-10_{s}')
    #
    # plt.figure(figsize=(50, 10))
    # plot_model_parcel(['Models_01/asym_Ib_space-MNISymC3_K-20',
    #                    'Models_02/asym_Ib_space-MNISymC3_K-20',
    #                    'Models_03/asym_Ib_space-MNISymC3_K-20',
    #                    'Models_04/asym_Ib_space-MNISymC3_K-20',
    #                    'Models_05/asym_Ib_space-MNISymC3_K-20'], [1,5], cmap='tab20',
    #                   align=True)
    # plt.savefig('ib_k-20_allsess.png', format='png')
    # plt.show()

    
    # ########## Higher K ##########
    space = 'MNISymC3' # Set atlas space
    # space = 'MNISymC2' # Set atlas space
    msym = 'sym' # Set model symmetry
    # ks = [10, 20, 34]
    ks = [14, 28, 48, 56, 60]
    # ks = [34, 40, 68, 80]
    # ks=[80]


    # # -- Build dataset list --
    n_dsets = 7 # without HCP
    alldatasets = np.arange(n_dsets).tolist()
    loo_datasets = [ np.delete(np.arange(n_dsets), d).tolist() for d in alldatasets ]
    individual_datasets = [ [d] for d in alldatasets ]

    dataset_list = []
    dataset_list.extend([alldatasets])
    dataset_list.extend(individual_datasets)
    # dataset_list.extend(loo_datasets)


    T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    for t in ['03']:
        for datasets in dataset_list:
            for k in ks:
                datanames = ''.join(T.two_letter_code[datasets])
                wdir = model_dir + f'/Models/Models_{t}'
                fname = f'/sym_{datanames}_space-{space}_K-{k}.tsv'

                if not Path(wdir+fname).exists():
                    print(f'fitting model {t} with K={k} as {fname}...')
                    fit_all(datasets, k, model_type=t, repeats=100, sym_type=[msym], space=space)
                else:
                    print(f'model {t} with K={k} already fitted as {fname}')

    # # # -- Build dataset list with HCP--
    # n_dsets = 8 # with HCP
    # alldatasets = np.arange(n_dsets).tolist()
    # loo_datasets = [ np.delete(np.arange(n_dsets), d).tolist() for d in alldatasets ]
    # individual_datasets = [ [d] for d in alldatasets ]

    # dataset_list = []
    # dataset_list.extend([alldatasets])
    # dataset_list.extend(individual_datasets)
    # # dataset_list.extend(loo_datasets)


    # T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    # for datasets in dataset_list:
    #     for t in ['03','04']:
    #         for k in ks:


    #             datanames = ''.join(T.two_letter_code[datasets])
    #             wdir = model_dir + f'/Models/Models_{t}'
    #             fname = f'/sym_{datanames}_space-{space}_K-{k}.tsv'

    #             if not Path(wdir+fname).exists():
    #                 print(f'fitting model {t} with K={k} as {fname}...')
    #                 fit_all(datasets, k, model_type=t, repeats=100, sym_type=[s])
    #             else:
    #                 print(f'model {t} with K={k} already fitted as {fname}')

    pass
