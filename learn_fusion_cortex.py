#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to fusion datasets on the cortical data

Created on 11/17/2022 at 2:16 PM
Author: dzhi
"""
# Script for importing the MDTB data set from super_cerebellum to general format.
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import Functional_Fusion.matrix as matrix
from scipy.linalg import block_diag
import nibabel as nb
import SUITPy as suit
import generativeMRF.full_model as fm
import generativeMRF.spatial as sp
import generativeMRF.arrangements as ar
import generativeMRF.emissions as em
import generativeMRF.evaluation as ev
from ProbabilisticParcellation.util import *
import torch as pt
from learn_mdtb import get_mdtb_parcel
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import pickle
from copy import deepcopy
import time

# pytorch cuda global flag
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# Find model directory to save model fitting results
model_dir = 'Y:\data\Cerebellum\ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/robabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/robabilisticParcellationModel'
if not Path(model_dir).exists():
    raise (NameError('Could not find model_dir'))

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise (NameError('Could not find base_dir'))

atlas_dir = base_dir + f'/Atlases'


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


def batch_fit(datasets, sess,
              type=None, cond_ind=None, part_ind=None, subj=None,
              atlas=None,
              K=10,
              arrange='independent',
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
        model_type (str): String indicating model_type
        datasets (list): List of dataset names to be used as training
        sess (list): List of list of sessions to be used for each
        type (list): List the type
        cond_ind (list): Name of the info-field that indicates the condition
        part_ind (list): Name of the field indicating independent partitions of the data
        subj (list, optional): _description_. Defaults to None
        atlas (Atlas): Atlas to be used. Defaults to None.
        K (int): Number of parcels. Defaults to 10.
        arrange (str): Type of arangement model. Defaults to 'independent'.
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

    # Build the model
    # Check for size of Atlas + whether symmetric
    if isinstance(atlas, (am.AtlasSurfaceSymmetric,
                          am.AtlasVolumeSymmetric)):
        P_arrange = atlas.Psym
        K_arrange = np.ceil(K / 2).astype(int)
    else:
        P_arrange = atlas.P
        K_arrange = K

    print(f'Building fullMultiModel {arrange} + {emission} for fitting...')
    # Initialize arrangement model
    if arrange == 'independent':
        ar_model = ar.ArrangeIndependent(K=K_arrange, P=P_arrange,
                                         spatial_specific=True,
                                         remove_redundancy=False)
    else:
        raise (NameError(f'unknown arrangement model:{arrange}'))

    # Initialize emission models
    em_models = []
    for j, ds in enumerate(data):
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

    # Make a full fusion model
    if isinstance(atlas, (am.AtlasSurfaceSymmetric,
                          am.AtlasVolumeSymmetric)):
        M = fm.FullMultiModelSymmetric(ar_model, em_models,
                                       atlas.indx_full, atlas.indx_reduced,
                                       same_parcels=False)
    else:
        M = fm.FullMultiModel(ar_model, em_models)

    # Step 5: Estimate the parameter thetas to fit the new model using EM

    # Somewhat hacky: Weight different datasets differently
    if weighting is not None:
        M.ds_weight = weighting  # Weighting for each dataset

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
    prior = pt.zeros((n_fits, K_arrange, P_arrange))
    for i in range(n_fits):
        print(f'Start fit: repetition {i} - {name}')
        # Copy the obejct (without data)
        m = deepcopy(M)
        # Attach the data
        m.initialize(data, subj_ind=subj_ind)

        m, ll, theta, U_hat, ll_init = m.fit_em_ninits(
            iter=n_iter,
            tol=0.01,
            fit_arrangement=True,
            n_inits=n_inits,
            first_iter=first_iter)
        info.loglik.at[i] = ll[-1].cpu().numpy() # Convert to numpy
        m.clear()
        models.append(m)

        # Align group priors
        if i == 0:
            indx = pt.arange(K_arrange)
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

        # Convergence: 1. must run enough repetitions (50);
        #              2. num_rep greater than threshold (10% of max_iter)
        if (i>50) and (num_rep >= int(n_fits*0.1)):
            break

    # Align the different models
    models = np.array(models, dtype=object)
    ev.align_models(models)

    return info, models


def fit_all(set_ind=[0, 1, 2, 3], K=10, repeats=100, model_type='01',
            sym_type=[0,1], subj_list=None, weighting=None, this_sess=None):
    # Data sets need to numpy arrays to allow indixing by list
    datasets = np.array(['Mdtb', 'Pontine', 'Nishimoto', 'Ibc', 'Hcp'],
                        dtype=object)
    sess = np.array(['all', 'all', 'all', 'all', 'all'],
                    dtype=object)
    if this_sess is not None:
        for i, idx in enumerate(set_ind):
            sess[idx] = this_sess[i]

    type = np.array(['CondHalf', 'TaskHalf', 'CondHalf', 'CondHalf', 'NetRun'],
                    dtype=object)

    cond_ind = np.array(['cond_num_uni', 'task_num',
                         'reg_id', 'cond_num_uni', 'reg_id'], dtype=object)
    part_ind = np.array(['half', 'half', 'half', 'half', 'half']
                        , dtype=object)

    # Make the atlas object
    ############## To be uncomment for cortical parcellation ##############
    # atlas_asym = am.get_atlas('fs32k', atlas_dir)
    # bm_name = ['cortex_left', 'cortex_right']
    # mask = []
    # for i, hem in enumerate(['L', 'R']):
    #     mask.append(atlas_dir + f'/tpl-fs32k/tpl-fs32k_hemi-{hem}_mask.label.gii')
    # atlas_sym = am.AtlasSurfaceSymmetric('fs32k', mask_gii=mask, structure=bm_name)
    # atlas = [atlas_asym, atlas_sym]
    #######################################################################
    mask = base_dir + '/Atlases/tpl-MNI152NLIn2009cSymC/tpl-MNISymC_res-3_gmcmask.nii'
    atlas = [am.AtlasVolumetric('MNISymC3', mask_img=mask),
             am.AtlasVolumeSymmetric('MNISymC3', mask_img=mask)]

    # Give a overall name for the type of model
    mname = ['asym', 'sym']

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
    dataname = [datasets[i][0:2] for i in set_ind]

    for i in sym_type:
        tic = time.perf_counter()
        name = mname[i] + '_' + ''.join(dataname)
        info, models = batch_fit(datasets[set_ind],
                                 sess=sess[set_ind],
                                 type=type[set_ind],
                                 cond_ind=cond_ind[set_ind],
                                 part_ind=part_ind[set_ind],
                                 subj=subj_list,
                                 atlas=atlas[i],
                                 K=K,
                                 name=name,
                                 n_inits=100,
                                 n_iter=200,
                                 n_rep=repeats,
                                 first_iter=30,
                                 join_sess=join_sess,
                                 join_sess_part=join_sess_part,
                                 uniform_kappa=uniform_kappa,
                                 weighting=weighting)

        # Save the fits and information
        wdir = model_dir + f'/Models/Models_{model_type}'
        fname = f'/{name}_space-{atlas[i].name}_K-{K}'

        if this_sess is not None:
            return wdir, fname, info, models

        if subj_list is not None:
            wdir = model_dir + f'/Models/Models_{model_type}/leaveNout'
            fname = f'/{name}_space-{atlas[i].name}_K-{K}'
            return wdir, fname, info, models

        info.to_csv(wdir + fname + '.tsv', sep='\t')
        with open(wdir + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)

        toc = time.perf_counter()
        print(f'Done Model fitting - {mname[i]}. Used {toc - tic:0.4f} seconds!')


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
        data = data.cpu().numpy()

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
                                                sym_type=[0],
                                                subj_list=[sub_list])
            fname = fname + f'_leave-{i}'
            info.to_csv(wdir + fname + '.tsv', sep='\t')
            with open(wdir + fname + '.pickle', 'wb') as file:
                pickle.dump(models, file)

def fit_indv_sess_IBC(model_type='01'):
    sess = DataSetIBC(base_dir + '/IBC').sessions
    for indv_sess in sess:
        wdir, fname, info, models = fit_all([3], 10,
                                            model_type=model_type,
                                            repeats=100,
                                            sym_type=[0],
                                            this_sess=[[indv_sess]])
        fname = fname + f'_{indv_sess}'
        info.to_csv(wdir + fname + '.tsv', sep='\t')
        with open(wdir + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)

def fit_two_IBC_sessions(sess1='clips4', sess2='rsvplanguage', model_type='04'):
    wdir, fname, info, models = fit_all([3], 10, model_type=model_type, repeats=50,
                                        sym_type=[0], this_sess=[['ses-'+sess1,
                                                                  'ses-'+sess2]])
    fname = fname + f'_ses-{sess1}+{sess2}'
    info.to_csv(wdir + '/IBC_sessFusion' + fname + '.tsv', sep='\t')
    with open(wdir + '/IBC_sessFusion' + fname + '.pickle', 'wb') as file:
        pickle.dump(models, file)

if __name__ == "__main__":
    # fit_all([0], 10, model_type='04', repeats=100, sym_type=[0])
    ########## Reliability map
    # rel, sess = reliability_maps(base_dir, 'IBC')
    # plt.figure(figsize=(25, 18))
    # plot_multi_flat(rel, 'MNISymC3', grid=(3, 5), dtype='func',
    #                 cscale=[-0.3, 0.7], colorbar=False, titles=sess)

    ########## IBC selected sessions fusion fit ##########
    # sess_1 = DataSetIBC(base_dir + '/IBC').sessions
    # sess_2 = DataSetIBC(base_dir + '/IBC').sessions
    # for s1 in sess_1:
    #     sess_2.remove(s1)
    #     for s2 in sess_2:
    #         this_s1 = s1.split('-')[1]
    #         this_s2 = s2.split('-')[1]
    #         wdir = model_dir + '\Models\Models_04\IBC_sessFusion'
    #         fname = wdir+f'/asym_Ib_space-MNISymC3_K-10_ses-{this_s1}+{this_s2}.tsv'
    #         if not os.path.isfile(fname):
    #             fit_two_IBC_sessions(sess1=this_s1, sess2=this_s2, model_type='04')
    #             print(f'-Done type 04 fusion {s1} and {s2}.')

    ########## IBC all sessions fit ##########
    # fit_indv_sess_IBC(model_type='03')
    # dataset_list = [[0], [1], [2], [3], [0,1,2,3]]

    ########## IBC all fit ##########
    type_list = ['02','03','04','05','01']
    K = [20, 34]
    for t in type_list:
        for k in K:
            fit_all([3], k, model_type=t, repeats=100, sym_type=[0])

    ########## Leave-one-oout ##########
    # leave_one_out_fit(dataset=dataset_list, model_type=type_list, K=10)

    ########## Plot the flatmap results ##########
    # Read the MDTB colors
    color_file = atlas_dir + '/tpl-SUIT/atl-MDTB10.lut'
    color_info = pd.read_csv(color_file, sep=' ', header=None)
    MDTBcolors = color_info.iloc[:, 1:4].to_numpy()

    # Make IBC session model file names
    # fnames = []
    # sess = DataSetIBC(base_dir + '/IBC').sessions
    # for s in sess:
    #     fnames.append(f'Models_05/asym_Ib_space-MNISymC3_K-10_{s}')

    plt.figure(figsize=(50, 10))
    plot_model_parcel(['Models_01/asym_Ib_space-MNISymC3_K-10',
                       'Models_02/asym_Ib_space-MNISymC3_K-10',
                       'Models_03/asym_Ib_space-MNISymC3_K-10',
                       'Models_04/asym_Ib_space-MNISymC3_K-10',
                       'Models_05/asym_Ib_space-MNISymC3_K-10'], [1,5], cmap=MDTBcolors,
                      align=True)
    plt.savefig('ib_k-10_allsess.png', format='png')
    plt.show()

    pass
