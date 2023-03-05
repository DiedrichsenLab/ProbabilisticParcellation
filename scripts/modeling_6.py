#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional result 6 integrating resting state vs. purely task

Created on 2/17/2023 at 11:40 AM
Author: dzhi
"""
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
from Functional_Fusion.dataset import *
import generativeMRF.emissions as em
import generativeMRF.arrangements as ar
import generativeMRF.full_model as fm
import generativeMRF.evaluation as ev

from scipy.linalg import block_diag
import nibabel as nb
import SUITPy as suit
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import time
import pickle
from copy import copy,deepcopy
from itertools import combinations
import ProbabilisticParcellation.util as ut
from ProbabilisticParcellation.evaluate import *
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.hierarchical_clustering as cl
from ProbabilisticParcellation.learn_fusion_gpu import *

# pytorch cuda global flag
# pt.cuda.is_available = lambda : False
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# Find model directory to save model fitting results
model_dir = 'Y:/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    raise (NameError('Could not find model_dir'))

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:/data/FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))

atlas_dir = base_dir + f'/Atlases'
res_dir = model_dir + f'/Results' + '/5.all_datasets_fusion'

def get_cmap(mname, load_best=True, sym=False):
    # Get model and atlas.
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    if load_best:
        info, model = load_batch_best(mname)
    else:
        info, model = load_batch_fit(mname)
    atlas, ainf = am.get_atlas(info.atlas, atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.marginal_prob())
    parcel = Prob.argmax(axis=0) + 1

    # Get parcel similarity:
    w_cos_sim, _, _ = cl.parcel_similarity(model, plot=False, sym=sym)
    W = sc.calc_mds(w_cos_sim, center=True)

    # Define color anchors
    m, regions, colors = sc.get_target_points(atlas, parcel)
    cmap = sc.colormap_mds(W, target=(m, regions, colors), clusters=None, gamma=0.3)

    return cmap.colors

def result_6_eval(model_name, K='10', t_datasets=['MDTB','Pontine','Nishimoto'],
                  out_name=None):
    """Evaluate group and individual DCBC and coserr of IBC single
       sessions on all other test datasets.
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
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        # Preparing atlas, cond_vec, part_vec
        if ds == 'HCP':
            this_type = 'Tseries'
            subj = np.arange(0, 100, 2)
            # tds = get_dataset_class(base_dir, 'HCP')
            # data_cereb, info = tds.get_data(space=space, ses_id=ses_id,
            #                                 type='Tseries', subj=[p])
            tic = time.perf_counter()
            tdata, tinfo, tds = get_dataset(base_dir, ds, atlas='MNISymC3',
                                            sess='all', type=this_type, subj=subj)
            toc = time.perf_counter()
            print(f'Done loading. Used {toc - tic:0.4f} seconds!')
        else:
            this_type = T.loc[T.name == ds]['default_type'].item()
            subj = None
            tic = time.perf_counter()
            tdata, tinfo, tds = get_dataset(base_dir, ds, atlas='MNISymC3',
                                            sess='all', type=this_type, subj=subj)
            toc = time.perf_counter()
            print(f'Done loading. Used {toc - tic:0.4f} seconds!')

        cond_vec = tinfo['time_id'].values.reshape(-1, ) # default from dataset class
        part_vec = tinfo['half'].values
        # part_vec = np.ones((tinfo.shape[0],), dtype=int)
        CV_setting = [('half', 1), ('half', 2)]

        ################ CV starts here ################
        atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')
        for (indivtrain_ind, indivtrain_values) in CV_setting:
            # get train/test index for cross validation
            train_indx = tinfo[indivtrain_ind] == indivtrain_values
            test_indx = tinfo[indivtrain_ind] != indivtrain_values
            # 1. Run DCBC individual
            res_dcbc = run_dcbc(model_name, tdata, atlas,
                               train_indx=train_indx,
                               test_indx=test_indx,
                               cond_vec=cond_vec,
                               part_vec=part_vec,
                               device='cuda')
            res_dcbc['indivtrain_ind'] = indivtrain_ind
            res_dcbc['indivtrain_val'] = indivtrain_values
            res_dcbc['test_data'] = ds
            results = pd.concat([results, res_dcbc], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    if out_name is None:
        fname = f'/eval_all_asym_K-{K}_on_MdHcEven.tsv'
    else:
        fname = f'/eval_all_asym_K-{K}_{out_name}_on_MdHcEven.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

def fit_rest_vs_task(datasets_list = [1,7], K=[34], sym_type=['asym'],
                     model_type=['03','04'], space='MNISymC3'):
    """Fitting model of task-datasets (MDTB out) + HCP (half subjects)

    Args:
        datasets_list: the dataset indices list
        K: number of parcels
        sym_type: atlas type
        model_type: fitting model types
        space: atlas space
    Returns:
        write in fitted model in .tsv and .pickle
    """
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    num_subj = T.return_nsubj.to_numpy()[datasets_list]
    sub_list = [np.arange(c) for c in num_subj[:-1]]

    # Odd indices for training, Even for testing
    hcp_train = np.arange(0, num_subj[-1], 2) + 1
    hcp_test = np.arange(0, num_subj[-1], 2)

    sub_list += [hcp_train]
    for t in model_type:
        for k in K:
            writein_dir = ut.model_dir + f'/Models/Models_{t}/leaveNout'
            dataname = ''.join(T.two_letter_code[datasets_list])
            nam = f'/asym_{dataname}_space-MNISymC3_K-{k}_hcpOdd'
            if not Path(writein_dir + nam + '.tsv').exists():
                wdir, fname, info, models = fit_all(set_ind=datasets_list, K=k,
                                                    repeats=100, model_type=t,
                                                    sym_type=sym_type,
                                                    subj_list=sub_list,
                                                    space=space)
                fname = fname + f'_hcpOdd'
                info.to_csv(wdir + fname + '.tsv', sep='\t')
                with open(wdir + fname + '.pickle', 'wb') as file:
                    pickle.dump(models, file)
            else:
                print(f"Already fitted {dataname}, K={k}, Type={t}...")

def plot_result_6(D, t_data='MDTB'):
    D = D.replace(["['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic' 'HCP']",
                   "['Pontine' 'Nishimoto' 'IBC' 'WMFS' 'Demand' 'Somatotopic']",
                   "['HCP']"], ['task+rest', 'task', 'rest'])
    D = D.loc[D.test_data == t_data]

    plt.figure(figsize=(10, 10))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(2, 2, i*2 + 1)
        sb.barplot(data=D, x='model_type', y=c, hue='train_data',
                   hue_order=['task','rest','task+rest'], errorbar="se")

        # if 'coserr' in c:
        #     plt.ylim(0.4, 1)
        plt.subplot(2, 2, i*2 + 2)
        sb.lineplot(data=D, x='K', y=c, hue='train_data',
                    hue_order=['task','rest','task+rest'],
                    style="model_type", errorbar='se', markers=False)

    plt.suptitle(f'Task, rest, task+rest, test_data={t_data}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ############# Fitting models #############
    for i in range(1,7):
        datasets_list = [0, 1, 2, 3, 4, 5, 6, 7]
        datasets_list.remove(i)
        print(datasets_list)
        fit_rest_vs_task(datasets_list=datasets_list, K=[34,40],
                         sym_type=['asym'], model_type=['04'], space='MNISymC3')

    ############# Evaluating models #############
    # model_type = ['03', '04']
    # K = [10,17,20,34,40,68,100]
    #
    # model_name = []
    # T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    # for i in range(0,1):
    # # for i in range(1, 7):
    #     datasets_list = [0, 1, 2, 3, 4, 5, 6]
    #     datasets_list.remove(i)
    #     dataname = ''.join(T.two_letter_code[datasets_list])
    #     # Pure Task
    #     model_name += [f'Models_{mt}/asym_{dataname}_space-MNISymC3_K-{this_k}'
    #                    for this_k in K for mt in model_type]
    #     # Task+rest
    #     model_name += [f'Models_{mt}/leaveNout/asym_{dataname}Hc_space-MNISymC3_K-{this_k}_hcpOdd'
    #                    for this_k in K for mt in model_type]
    #
    # # Pure Rest
    # model_name += [f'Models_{mt}/leaveNout/asym_Hc_space-MNISymC3_K-{this_k}_hcpOdd'
    #                for this_k in K for mt in model_type]
    #
    # result_6_eval(model_name, K='10to100', t_datasets=['HCP'], out_name='6taskHcOdd')
    #
    # ############# Plot evaluation #############
    # fname = f'/Models/Evaluation/eval_all_asym_K-10to100_6taskHcOdd_on_HcEven_ts.tsv'
    # D = pd.read_csv(model_dir + fname, delimiter='\t')
    # plot_result_6(D, t_data='HCP')

    ############# Plot fusion atlas #############
    # Making color map
    # K = 34
    # fname = [f'/Models_03/asym_PoNiIbWmDeSo_space-MNISymC3_K-{K}',
    #          f'/Models_03/leaveNout/asym_Hc_space-MNISymC3_K-{K}_hcpOdd',
    #          f'/Models_03/leaveNout/asym_PoNiIbWmDeSoHc_space-MNISymC3_K-{K}_hcpOdd']
    # colors = get_cmap(f'/Models_03/asym_PoNiIbWmDeSo_space-MNISymC3_K-{K}')
    #
    # plt.figure(figsize=(20, 10))
    # plot_model_parcel(fname, [1, 3], cmap=colors, align=True, device='cuda')
    # plt.show()