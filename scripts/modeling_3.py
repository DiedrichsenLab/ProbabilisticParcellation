#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result 3: IBC two sessions fusion

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
res_dir = model_dir + f'/Results' + '/3.IBC_two_sessions'

def result_3_eval(K=10, model_type=['03','04'], ses1=None, ses2=None):
    """Result3: Evaluate group and individual DCBC and coserr
       of IBC two single sessions and fusion on the IBC
       left-out sessions.
       i.e [sess1, sess2, sess1and2 fusion]
    Args:
        ses1: the first session. i.e 'ses-archi'
        ses2: the second session. i.e 'ses-archi'
    Returns:
        Write in evaluation file
    Notes:
        if ses1 and ses2 are both None, the function performs
        all 91 combination of IBC two sessions fusion.
    """
    # Calculate the session reliability value
    rel, sess = reliability_maps(base_dir, 'IBC', subtract_mean=False,
                                 voxel_wise=False)
    reliability = dict(zip(sess, rel[:, 0]))
    sess_1 = DataSetIBC(base_dir + '/IBC').sessions
    sess_2 = DataSetIBC(base_dir + '/IBC').sessions
    if (ses1 is not None) and (ses2 is not None):
        sess_1 = [ses1, ses2]
        sess_2 = [ses1, ses2]

    results = pd.DataFrame()
    for s1 in sess_1:
        sess_2.remove(s1)
        for s2 in sess_2:
            this_s1 = s1.split('-')[1]
            this_s2 = s2.split('-')[1]
            print(f'- Start evaluating {this_s1} and {this_s2}.')
            # Making the models for both common/separate kappa
            model_name = []
            for mt in model_type:
                model_name += [f'Models_{mt}/asym_Ib_space-MNISymC3_K-{K}_ses-{this_s1}',
                               f'Models_{mt}/asym_Ib_space-MNISymC3_K-{K}_ses-{this_s2}',
                               f'Models_{mt}/IBC_sessFusion/asym_Ib_space-MNISymC3_K-{K}_'
                               f'ses-{this_s1}+{this_s2}']

            # remove the sessions were used to training to make test sessions
            this_sess = [i for i in sess if i not in [s1, s2]]
            # 1. Run DCBC individual
            res_dcbc = run_dcbc_individual(model_name, 'IBC', this_sess, cond_ind=None,
                                           part_ind=None, indivtrain_ind=None,
                                           indivtrain_values=[0], device='cuda')
            # 2. Run coserr individual
            res_coserr = run_prederror(model_name, 'IBC', this_sess, cond_ind=None,
                                       part_ind=None, eval_types=['group', 'floor'],
                                       indivtrain_ind=None, indivtrain_values=[0],
                                       device='cuda')
            # 3. Merge the two dataframe
            res = pd.merge(res_dcbc, res_coserr, how='outer')
            res['sess1_rel'] = reliability[s1]
            res['sess2_rel'] = reliability[s2]
            res['test_sess_out'] = this_s1 + '+' + this_s2
            results = pd.concat([results, res], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    if (ses1 is not None) and (ses2 is not None):
        fname = f'/eval_all_asym_Ib_K-{K}_{ses1}+{ses2}_on_leftSess.tsv'
    else:
        fname = f'/eval_all_asym_Ib_K-{K}_twoSess_on_leftSess.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

def result_3_plot(fname, train_model='IBC', ck=None, style=None, style_order=None,
                  relevant=None, print_relevancy=False):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    D['relevant'] = ""
    # D.rename(columns={'test_sess': 'session'}, inplace=True)

    # D = D.loc[D['K']==17]
    if ck is not None:
        D = D.loc[D['common_kappa']==ck]

    # 1. Get session-similarity between all 14 seesions
    rel, sess = reliability_maps(base_dir, train_model, subtract_mean=False,
                                 voxel_wise=True)
    ses_cor = np.corrcoef(rel)

    # 2. Calculate within-session reliability
    rel_ses, _ = reliability_maps(base_dir, train_model, subtract_mean=False,
                                  voxel_wise=False)
    # np.fill_diagonal(ses_cor, 0)
    # rel.sort(axis=1)
    # reliability = dict(zip(sess, rel[:,-int(rel.shape[1] * 0.5):].mean(axis=1)))
    reliability = dict(zip(sess, rel_ses[:, 0]))

    T = pd.DataFrame()
    for s1,s2 in combinations(sess, 2):
        df = D.loc[(D['test_sess_out'] == s1.split('-')[1] + '+' + s2.split('-')[1])]
        # if reliability[s1] >= reliability[s2]:
        #     df.loc[df.session == s1.split('-')[1], 'session'] = 'sess_1'
        #     df.loc[df.session == s2.split('-')[1], 'session'] = 'sess_2'
        # else:
        #     df.loc[df.session == s1.split('-')[1], 'session'] = 'sess_2'
        #     df.loc[df.session == s2.split('-')[1], 'session'] = 'sess_1'
        df.loc[df.session == s1.split('-')[1], 'session'] = 'sess_1'
        df.loc[df.session == s2.split('-')[1], 'session'] = 'sess_2'

        df.loc[df.session == s1.split('-')[1] + '+' + s2.split('-')[1], 'session'] = 'Fusion'
        # Pick the two sessions with larger reliability difference
        # if abs(reliability[s1] - reliability[s2]) > 0.3 * dif_rel:
        # if (reliability[s1] > mean_rel and reliability[s2] < mean_rel) or \
        #         (reliability[s1] < mean_rel and reliability[s2] > mean_rel):

        ###### Triage current two sessions into relevant/irrelevant ######
        # Note: this triage can be modified by manually select which two
        # sessions are relevant or irrelevant. Right now, it is triaged by
        # the between-session similarity (<0 indicats irrelevant; >0.4 means
        # relevant sessions)
        ##################################################################
        if ses_cor[sess.index(s1)][sess.index(s2)] <= 0:
            if print_relevancy:
                print(f'irrelevant sessions: {s1} and {s2}')
            df.loc[df.index, 'relevant'] = False
        elif ses_cor[sess.index(s1)][sess.index(s2)] >= 0.4:
            if print_relevancy:
                print(f'relevant sessions: {s1} and {s2}')
            df.loc[df.index, 'relevant'] = True
        else:
            df.loc[df.index, 'relevant'] = 'Between'

        T = pd.concat([T, df], ignore_index=True)

    if relevant is not None:
        T = T.loc[T.relevant == relevant]

    plt.figure(figsize=(8,5))
    crits = ['dcbc_group','dcbc_indiv']
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        # sb.barplot(data=T, x='session', y=c, order=['sess_1','sess_2','Fusion'], hue='model_type',
        #            hue_order=T['model_type'].unique(), errorbar="se")
        if style is not None:
            sb.lineplot(data=T, x="K", y=c, hue='session', hue_order=['sess_1','sess_2','Fusion'],
                        style=style, style_order=style_order, markers=True)
        else:
            sb.lineplot(data=T, x="K", y=c, hue='session',
                        hue_order=['sess_1','sess_2','Fusion'], markers=True)
        # if c == 'dcbc_indiv':
        #     plt.ylim(0, 0.04)
        # elif c == 'dcbc_group':
        #     plt.ylim(0, 0.04)
        # elif c== 'coserr_floor':
        #     plt.ylim(0.475, 0.525)

        # plt.legend(loc='lower right')

    plt.suptitle(f'IBC two sessions fusion - overall trend (two sessions have overlapping)')
    plt.tight_layout()
    plt.savefig('Ibc_twoSessFusion.pdf', format='pdf')
    plt.show()

def result_3_rel_check(fname, K=[10,17,20,34,40,68], save=False):
    D_all = pd.read_csv(model_dir + fname, delimiter='\t')
    # Calculate session reliability
    rel, sess = reliability_maps(base_dir, 'IBC', subtract_mean=False,
                                 voxel_wise=False)
    reliability = dict(zip(sess, rel.mean(axis=1)))
    crits = ['dcbc_group', 'dcbc_indiv', 'coserr_group', 'coserr_floor']

    row = len(K)
    col = len(crits)
    plt.figure(figsize=(col*5, row*5))
    for j, k in enumerate(K):
        D = D_all.loc[D_all.K == k]
        for i, c in enumerate(crits):
            plt.subplot(row, col, j*col+i+1)
            T = pd.DataFrame()
            for s in sess:
                this_df = D.loc[(D['session'] == s.split('-')[1])].reset_index()
                this_df['reliability'] = reliability[s]
                T = pd.concat([T, this_df], ignore_index=True)
                for ck in [True, False]:
                    plt.text(reliability[s], this_df.loc[(this_df['common_kappa'] == ck), c].mean(),
                             s.split('-')[1], fontdict=dict(color='black', alpha=0.5))

            sb.lineplot(data=T, x="reliability", y=c, hue="common_kappa",
                        hue_order=T['common_kappa'].unique(), errorbar="se",
                        err_style="bars", markers=True, markersize=10)
            # sb.scatterplot(data=T, x="reliability", y=c, hue="common_kappa",
            #                hue_order=T['common_kappa'].unique())

    plt.suptitle(f'IBC individual sessions perfermance vs reliability, '
                 f'test_data=IBC_leftoutSess, K={K}')
    plt.tight_layout()

    if save:
        plt.savefig('IBC_sess_performance_reliability_check.pdf', format='pdf')
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

def plot_IBC_performance_reliability(K=[10,17,20,34,40,68], save=False):
    fname1 = f'/Models/Evaluation/eval_all_asym_Ib_K-10_to_68_indivSess_on_otherDatasets.tsv'
    fname2 = f'/Models/Evaluation/eval_asym_train-Ib_twoSess_test-leftOutSess.tsv'
    D1_all = pd.read_csv(model_dir + fname1, delimiter='\t')
    D2_all = pd.read_csv(model_dir + fname2, delimiter='\t')

    # Uniform the coserr individual column name in the two files
    D1_all.rename(columns={'coserr_ind3': 'coserr_indiv'}, inplace=True)
    D2_all.rename(columns={'coserr_floor': 'coserr_indiv'}, inplace=True)
    sess = DataSetIBC(base_dir + '/IBC').sessions

    crits = ['dcbc_group', 'dcbc_indiv', 'coserr_group', 'coserr_indiv']
    num_row = len(K)
    num_col = len(crits)
    plt.figure(figsize=(num_col*5, num_row*5))
    for i, k in enumerate(K):
        D1 = D1_all.loc[D1_all.K == k]
        D2 = D2_all.loc[D2_all.K == k]
        for j, c in enumerate(crits):
            plt.subplot(num_row, num_col, i*num_col+j+1)
            T = pd.DataFrame()

            for s in sess:
                this_df = D1.loc[D1.session == s.split('-')[1]]
                this_df2 = D2.loc[D2.session == s.split('-')[1]]
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

    plt.suptitle(f'IBC sessions performance consistancy, '
                 f'tested on otherDatasets vs. leftOutSess, K={K}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        plt.savefig('IBC_sess_performance_consistancy_check.pdf', format='pdf')
    plt.show()

def make_all_in_one_tsv(path, out_name):
    """Making all-in-one tsv file of evaluation
    Args:
        path: the path of the folder that contains
              all tsv files will be integrated
        out_name: output file name
    Returns:
        None
    """
    files = os.listdir(path)

    if not any(".tsv" in x for x in files):
        raise Exception('Input data file type must be .tsv file!')
    else:
        D = pd.DataFrame()
        for fname in files:
            res = pd.read_csv(path + f'/{fname}', delimiter='\t')

            # Making sure <PandasArray> mistakes are well-handled
            trains = res["train_data"].unique()
            print(trains)
            D = pd.concat([D, res], ignore_index=True)

        D.to_csv(out_name, sep='\t', index=False)

if __name__ == "__main__":
    ##### 1. Evaluate all two sessions fusion tested on 12 leftout sessions
    ##### The number of combination = 91 (pick 2 from 14)
    for k in [17]:
        result_3_eval(K=k, model_type=['06'])

    # make_all_in_one_tsv('Y:\data\Cerebellum\ProbabilisticParcellationModel\Models\Evaluation_01',
    #                     'Y:\data\Cerebellum\ProbabilisticParcellationModel\Models\Evaluation_01'
    #                     '\model1.tsv')
    # fname = '/Models/Evaluation_01/model1.tsv'
    fname = f'/Models/Evaluation/eval_asym_train-Ib_twoSess_test-leftOutSess.tsv'
    # result_3_plot(fname, style='common_kappa', style_order=[True, False], relevant=None)
    # ##### 2. Check whether the session perfermance is related to reliability
    # ##### The answer is No! (IBC sessions performance unrelated to reliability)
    # # result_3_rel_check(fname, K=[10,17,20,34,40,68])
    #
    # ##### 3. Further check whether the session perfermance is consistant tested on
    # ##### leftout sessions or tested on other clean datasets
    # ##### The answer is Yes! (IBC sessions performance is consistant across testsets)
    # plot_IBC_performance_reliability(K=[10,17,20,34,40,68], save=True)
    #
    # ##### 4. Plot sess-1, sess-2, Fusion (indiv/group DCBC and coserr)
    # # Option 1: overall trend (ignoring relevant/irrelevant sessions)
    result_3_plot(fname, ck=None, style='model_type',
                  style_order=['Models_01','Models_03','Models_04'], relevant=None)
    # # Option 2: overall trend (triaged by relevant/irrelevant sessions)
    # result_3_plot(fname, ck=True, style='relevant', style_order=[True, False])

    # ##### Plot the indiv and fusion map #####
    # colors = get_cmap('Models_03/IBC_sessFusion/asym_Ib_space-MNISymC3_K-17_ses-preference+tom')
    # colors[-1] = colors[1]
    # colors[1] = np.array([254 / 255, 254 / 255, 0 / 255, 1.])
    # colors[5] = np.array([232 / 255, 114 / 255, 232 / 255, 1.])
    # plot_model_parcel(['Models_03/IBC_sessFusion/asym_Ib_space-MNISymC3_K-17_ses-preference+tom',
    #                    'Models_03/asym_Ib_space-MNISymC3_K-17_ses-preference',
    #                    'Models_03/asym_Ib_space-MNISymC3_K-17_ses-tom'],
    #                    [1, 3], cmap=colors, align=True, device='cuda')

    # colors = get_cmap('Models_04/IBC_sessFusion/asym_Ib_space-MNISymC3_K-17_ses-preference+tom')
    # plot_model_parcel(['Models_04/IBC_sessFusion/asym_Ib_space-MNISymC3_K-17_ses-preference+tom',
    #                    'Models_04/asym_Ib_space-MNISymC3_K-17_ses-preference',
    #                    'Models_04/asym_Ib_space-MNISymC3_K-17_ses-tom'],
    #                    [1, 3], cmap=colors, align=True, device='cuda')
    # plt.show()

    ##### Plot IBC session reliability map #####
    # rel, sess = reliability_maps(base_dir, 'IBC', subtract_mean=False, voxel_wise=True)
    # plot_multi_flat(rel, 'MNISymC3', grid=(3, 5), dtype='func',
    #                 cscale=[-0.3, 0.7], colorbar=False, titles=sess)