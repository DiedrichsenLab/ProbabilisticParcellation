#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result 4: IBC individual session vs. all sessions fusion

Created on 1/5/2023 at 11:14 AM
Author: dzhi
"""
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
from Functional_Fusion.dataset import *
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as ev

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
from ProbabilisticParcellation.util import *
from ProbabilisticParcellation.evaluate import *

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
res_dir = model_dir + f'/Results'

def result_4_eval(K=[10], model_type=['03','04'], t_datasets=['MDTB','Pontine','Nishimoto'],
                  test_ses=None):
    """Evaluate group and individual DCBC and coserr of IBC single
       sessions on all other test datasets.
    Args:
        K: the number of parcels
    Returns:
        Write in evaluation file
    """
    sess = DataSetIBC(base_dir + '/IBC').sessions
    if test_ses is not None:
        sess = [test_ses]

    model_name = []
    # Making all IBC indiv sessions list
    for mt in model_type:
        model_name += [f'Models_{mt}/asym_Ib_space-MNISymC3_K-{this_k}_{s}'
                       for this_k in K for s in sess]

        # Additionally, add all IBC sessions fusion to list
        model_name += [f'Models_{mt}/asym_Ib_space-MNISymC3_K-{this_k}'
                       for this_k in K]

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')

        # Preparing atlas, cond_vec, part_vec
        atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')
        tdata, tinfo, tds = get_dataset(base_dir, ds, atlas='MNISymC3', sess='all')
        cond_vec = tinfo[tds.cond_ind].values.reshape(-1, ) # default from dataset class
        part_vec = tinfo['half'].values
        # part_vec = np.ones((tinfo.shape[0],), dtype=int)
        CV_setting = [('half', 1), ('half', 2)]

        ################ CV starts here ################
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
            # 2. Run coserr individual
            # res_coserr = run_prederror(model_name, ds, 'all', cond_ind=None,
            #                            part_ind='half', eval_types=['group', 'floor'],
            #                            indivtrain_ind='half', indivtrain_values=[1,2],
            #                            device='cuda')
            # 3. Merge the two dataframe
            # res = pd.merge(res_dcbc, res_coserr, how='outer')
            results = pd.concat([results, res_dcbc], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation_01'
    if test_ses is not None:
        fname = f'/eval_all_asym_Ib_K-{K}_{test_ses}_on_otherDatasets.tsv'
    else:
        fname = f'/eval_all_asym_Ib_K-10_to_100_indivSess_on_otherDatasets.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

def result_4_plot(fname, test_data=None, orderby=None, ck=True):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    if test_data is not None:
        D = D.loc[(D['test_data'] == test_data)]
    else:
        test_data = 'all'

    plt.figure(figsize=(15,15))
    crits = ['dcbc_group','dcbc_indiv','coserr_group','coserr_floor']
    for i, c in enumerate(crits):
        plt.subplot(4, 1, i + 1)
        if orderby is not None:
            order = D.loc[(D['common_kappa'] == orderby)].groupby('session')[c].mean().sort_values(
            ).keys().to_list()
        else:
            order = D.groupby('session')[c].mean().sort_values().keys().to_list()
        sb.barplot(data=D, x='session', y=c, order=order, hue='common_kappa',
                   hue_order=D.common_kappa.unique(), errorbar="se")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
        if c == 'coserr_group':
            plt.ylim(0.8, 0.9)
        if c == 'coserr_floor':
            plt.ylim(0.4, 0.6)

    plt.suptitle(f'IBC individual sessions vs. all sessions fusion, test_data={test_data}')
    plt.show()

def result_4_rel_check(fname, train_model='IBC', t_data=['Mdtb']):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    # Calculate session reliability
    rel, sess = reliability_maps(base_dir, train_model, subtract_mean=False,
                                 voxel_wise=False)
    reliability = dict(zip(sess, rel.mean(axis=1)))
    crits = ['dcbc_group', 'dcbc_indiv', 'coserr_group', 'coserr_floor']
    num_row = len(t_data)
    num_col = len(crits)

    plt.figure(figsize=(25, 15))
    for i, td in enumerate(t_data):
        df = D.loc[(D['test_data'] == td)]
        for j, c in enumerate(crits):
            plt.subplot(num_row, num_col, i*num_col+j+1)
            T = pd.DataFrame()
            for s in sess:
                this_df = df.loc[df.D == s.split('-')[1]].reset_index()
                this_df['reliability'] = reliability[s]
                this_df['session'] = s
                T = pd.concat([T, this_df], ignore_index=True)
                for ck in [True, False]:
                    plt.text(reliability[s], this_df.loc[(this_df['common_kappa'] == ck), c].mean(),
                             s.split('-')[1], fontdict=dict(color='black', alpha=0.5))

            sb.lineplot(data=T, x="reliability", y=c, hue="common_kappa",
                        hue_order=T['common_kappa'].unique(),errorbar="se",
                        err_style="bars", markers=True, markersize=10)
            # sb.scatterplot(data=T, x="reliability", y=c, hue="common_kappa",
            #                hue_order=T['common_kappa'].unique())

    plt.suptitle(f'{train_model} individual sessions perfermance vs reliability, test_data={t_data}')
    plt.show()

def plot_diffK(fname, hue="test_data", style="common_kappa"):
    D = pd.read_csv(model_dir + fname, delimiter='\t')

    df = D.loc[(D['model_type'] == 'Models_03')|(D['model_type'] == 'Models_04')]

    plt.figure(figsize=(12,15))
    crits = ['dcbc_group','dcbc_indiv','coserr_group',
             'coserr_floor','coserr_ind2','coserr_ind3']
    for i, c in enumerate(crits):
        plt.subplot(3, 2, i + 1)
        sb.lineplot(data=df, x="K", y=c, hue=style,
                    style_order=D[style].unique(),markers=True)
        # if i == len(crits)-1:
        #     plt.legend(loc='upper left')
        # else:
        #     plt.legend('')
        # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')

    plt.suptitle(f'All datasets fusion, diff K = {df.K.unique()}')
    plt.tight_layout()
    plt.show()

def plot_IBC_rel():
    fname1 = f'/Models/Evaluation/eval_all_asym_Ib_K-10_indivSess_on_otherDatasets.tsv'
    fname2 = f'/Models/Evaluation/eval_all_asym_Ib_K-10_twoSess_on_leftSess.tsv'
    D1 = pd.read_csv(model_dir + fname1, delimiter='\t')
    D2 = pd.read_csv(model_dir + fname2, delimiter='\t')
    sess = DataSetIBC(base_dir + '/IBC').sessions

    crits = ['dcbc_group', 'dcbc_indiv', 'coserr_group', 'coserr_floor']
    plt.figure(figsize=(15, 15))
    for j, c in enumerate(crits):
        plt.subplot(2, 2, j+1)
        T = pd.DataFrame()

        if c == 'coserr_floor':
            D1.rename(columns={'coserr_floor': 'coserr_ind3',
                               'coserr_ind3': 'coserr_floor'}, inplace=True)

        for s in sess:
            this_df = D1.loc[D1.D == s.split('-')[1]]
            this_df2 = D2.loc[D2.D == s.split('-')[1]]
            T1 = pd.DataFrame({'session': [s],
                               c+'_otherdata': [this_df.loc[(this_df['common_kappa'] == True),
                                                       c].mean()],
                               c+'_leftsess': [this_df2.loc[(this_df2['common_kappa'] == True),
                                                       c].mean()],
                               'common_kappa': [True]})
            T2 = pd.DataFrame({'session': [s],
                               c+'_otherdata': [this_df.loc[(this_df['common_kappa'] == False),
                                                       c].mean()],
                               c+'_leftsess': [this_df2.loc[(this_df2['common_kappa'] == False),
                                                       c].mean()],
                               'common_kappa': [False]})
            T = pd.concat([T, T1, T2], ignore_index=True)
            for ck in [True, False]:
                plt.text(this_df.loc[(this_df['common_kappa'] == ck), c].mean(),
                         this_df2.loc[(this_df2['common_kappa'] == ck), c].mean(),
                         s.split('-')[1], fontdict=dict(color='black', alpha=0.5))

        # sb.lineplot(data=T, x=c+'_otherdata', y=c+'_leftsess', hue="common_kappa",
        #             hue_order=T['common_kappa'].unique(),markers=True, markersize=10)
        sb.scatterplot(data=T, x=c+'_otherdata', y=c+'_leftsess', hue="common_kappa",
                       hue_order=T['common_kappa'].unique(), s=50)

    plt.suptitle(f'IBC individual sessions performance, tested on otherData vs. leftSess')
    plt.show()

def plot_fig1(fname, tdata=None, save=False):
    D = pd.read_csv(model_dir + fname, sep='\t')
    orderby = None

    if tdata is not None:
        D = D.loc[D['test_data'] == tdata]

    plt.figure(figsize=(12, 6))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(2, 1, i + 1)
        if orderby is not None:
            order = D.loc[(D['common_kappa'] == True)].groupby('session')[
                c].mean().sort_values().keys().to_list()
        else:
            order = D.groupby('test_sess')[c].mean().sort_values().keys().to_list()
        sb.barplot(data=D, x='test_sess', y=c, order=order, hue='model_type',
                   width=0.7, errorbar="se",
                   palette=sb.color_palette()[0:3])

        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize='small')
        # if c == 'dcbc_group':
        #     plt.ylim(0, 0.09)
        # elif c == 'dcbc_indiv':
        #     plt.ylim(0.1, 0.22)

    plt.suptitle(f'IBC individual sessions vs. all sessions fusion (averaged across Ks)')
    plt.tight_layout()

    if save:
        plt.savefig('IBC_indviSess_vs_fusion.pdf', format='pdf')
    plt.show()

def plot_fig2(save=False):
    D = pd.read_csv(model_dir +
                    '/Models/Evaluation_01/eval_all_asym_Ib_K-10_to_100_indivSess_on_otherDatasets'
                    '.tsv', sep='\t')

    plt.figure(figsize=(10, 5))
    crits = ['dcbc_group', 'dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        sb.lineplot(data=D, x='K', y=c, hue='model_type',
                    style='test_sess', errorbar="se",
                    palette=sb.color_palette()[0:3], err_style=None)
        if i == len(crits) - 1:
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,
                       fontsize='small')
        else:
            plt.legend('')

        if c == 'coserr_group':
            plt.ylim(0.7, 1.0)
    #     if c == 'coserr_floor':
    #         plt.ylim(0.4, 0.8)

    plt.suptitle(f'IBC individual sessions vs. all sessions fusion (varying across Ks)')
    plt.tight_layout()

    if save:
        plt.savefig('IBC_indviSess_vs_fusion_diffK.pdf', format='pdf')
    plt.show()


if __name__ == "__main__":
    # result_4_eval(K=[10,17,20,34,40,68,100], model_type=['01','03','04'],
    #               t_datasets=['MDTB', 'Pontine', 'Nishimoto',
    #                           'WMFS', 'Demand', 'Somatotopic'])
    # result_4_eval(K=[17], model_type=['01'], t_datasets=['MDTB'])

    # fname = f'/Models/Evaluation/eval_all_asym_Ib_K-10_to_100_indivSess_on_otherDatasets.tsv'
    # result_4_plot(fname, test_data='Pontine', orderby=False)
    fname = '/Models/Evaluation_01/eval_all_asym_Ib_K-10_to_100_indivSess_on_otherDatasets.tsv'
    plot_fig1(fname, save=False)
    plot_fig2(save=True)

    ######## Plot indiv sess vs fusion map ########
    # color_file = atlas_dir + '/tpl-SUIT/atl-NettekovenSym34.lut'
    # color_info = pd.read_csv(color_file, sep=' ', header=None)
    # colors = color_info.iloc[:,1:4].to_numpy()
    #
    # plt.figure(figsize=(20, 12))
    # sess = DataSetIBC(base_dir + '/IBC').sessions
    # model_names = [f'Models_03/asym_Ib_space-MNISymC3_K-34_{s}' for s in sess]
    # plot_model_parcel(model_names + [f'Models_03/asym_Ib_space-MNISymC3_K-34'],
    #                   [3, 5], cmap=colors, align=True, device='cuda')
    # plt.show()