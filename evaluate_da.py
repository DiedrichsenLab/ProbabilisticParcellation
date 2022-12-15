#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripts for functional fusion model paper ONLY!
It may not capatible with other major functions, use with caution!

Created on 12/13/2022 at 12:39 PM
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
from itertools import combinations
from ProbabilisticParcellation.util import *
from ProbabilisticParcellation.evaluate import *

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
    raise(NameError('Could not find base_dir'))

atlas_dir = base_dir + f'/Atlases'

def eval_generative_SNMF(model_names = ['asym_Md_space-SUIT3_K-10']):
    """This is the evaluation case of the parcellation comparison
    between the new fusion model vs. convex semi non-negative matrix
    factorization (King et. al, 2019).

    Args:
        model_names (list): the list of model names to be evaluated
    Returns:
        the plot
    Note:
        I'm just simply curious whether the fusion model fit on mdtb
        standalone (indepent ar. + VMF em.) can beat NMF algorithm
        or not. So nothing hurt the script.    -- dzhi
    """
    # Use specific mask / atlas.
    mask = base_dir + '/Atlases/tpl-SUIT/tpl-SUIT_res-3_gmcmask.nii'
    atlas = am.AtlasVolumetric('SUIT3', mask_img=mask)

    # get original mdtb parcels (nmf)
    from learn_mdtb import get_mdtb_parcel
    mdtb_par, _ = get_mdtb_parcel(do_plot=False)
    mdtb_par = np.where(mdtb_par == 0, np.nan, mdtb_par)

    parcel = np.empty((len(model_names), atlas.P))

    for i, mn in enumerate(model_names):
        info, models, Prop, V = load_batch_fit(mn)
        j = np.argmax(info.loglik)
        # Get winner take all
        par = pt.argmax(Prop[j, :, :], dim=0) + 1
        parcel[i, :] = np.where(np.isnan(mdtb_par), np.nan, par.numpy())

    # Evaluate case: use all MDTB data
    # It kinda of overfitting but still fair comparison
    data_eval, _, _ = get_all_mdtb(atlas='SUIT3')
    dcbc_base = eval_dcbc(mdtb_par, atlas, func_data=data_eval,
                          resolution=3, trim_nan=True)

    dcbc_compare = []
    for p in range(parcel.shape[0]):
        this_dcbc = eval_dcbc(parcel[p], atlas, func_data=data_eval,
                              resolution=3, trim_nan=True)
        dcbc_compare.append(this_dcbc)

    plt.figure()
    plt.bar(['NMF', 'ind+vmf'], [dcbc_base.mean(), dcbc_compare[0].mean()],
            yerr=[dcbc_base.std() / np.sqrt(24),
                  dcbc_compare[0].std() / np.sqrt(24)])
    plt.show()

def concat_all_prederror(model_type,prefix,K,outfile):
    D = pd.DataFrame()
    for p in prefix:
        for k in K:
            fname = base_dir + f'/Models/Evaluation_{model_type}/eval_prederr_{p}_K-{k}.tsv'
            T = pd.read_csv(fname,delimiter='\t')
            T['prefix'] = [p]*T.shape[0]
            D = pd.concat([D,T],ignore_index=True)
    oname = base_dir + f'/Models/Evaluation_{model_type}/eval_prederr_{outfile}.tsv'
    D.to_csv(oname,index=False,sep='\t')

    pass

def run_ibc_sessfusion_group_dcbc(sess1='preference', sess2='rsvplanguage'):
    ############# Evaluating IBC dataset sessions and plot results #############
    # remove the sessions were used to training to make test sessions
    sess = DataSetIBC(base_dir + '/IBC').sessions
    sess.remove('ses-' + sess1)
    sess.remove('ses-' + sess2)

    # Run DCBC group evaluation of the sess1, sess2, sess12fusion
    model_name = [f'Models_02/asym_Ib_space-MNISymC3_K-10_ses-{sess1}.pickle',
                  f'Models_02/asym_Ib_space-MNISymC3_K-10_ses-{sess2}.pickle',
                  f'Models_04/asym_Ib_space-MNISymC3_K-10_ses-{sess1}+{sess2}.pickle']
    res = run_dcbc_group(model_name, 'MNISymC3', 'IBC', test_sess=sess,
                         saveFile=f'Evaluation/eval_Ib_{sess1}+{sess2}_dcbc_group_on_leftsessions')
    sb.barplot(x='model_type', y='dcbc', data=res)
    plt.show()


def result_1_eval(model_name='Models_05/asym_Md_space-MNISymC3_K-10'):
    info, model = load_batch_best(model_name)
    # Individual training dataset:
    idata, iinfo, ids = get_dataset(base_dir, 'Mdtb', atlas='MNISymC3',
                                    sess=['ses-s1'], type='CondRun')

    # Test data set:
    tdata, tinfo, tds = get_dataset(base_dir, 'Mdtb', atlas='MNISymC3',
                                    sess=['ses-s2'], type='CondHalf')

    # convert tdata to tensor
    idata = pt.tensor(idata, dtype=pt.get_default_dtype())
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

    # Capatible with old model fitting
    if not hasattr(model.arrange, 'tmp_list'):
        if model.arrange.__class__.__name__ == 'ArrangeIndependent':
            model.arrange.tmp_list = ['estep_Uhat']
        elif model.arrange.__class__.__name__ == 'cmpRBM':
            model.arrange.tmp_list = ['epos_Uhat', 'epos_Hhat',
                                      'eneg_U', 'eneg_H']

    if not hasattr(model.emissions[0], 'tmp_list'):
        if model.emissions[0].name == 'VMF':
            model.emissions[0].tmp_list = ['Y', 'num_part']
        elif model.emissions[0].name == 'wVMF':
            model.emissions[0].tmp_list = ['Y', 'num_part', 'W']

    # Build the individual training model on session 1:
    m1 = deepcopy(model)
    cond_vec = iinfo['cond_num_uni'].values.reshape(-1, )
    part_vec = iinfo['run'].values.reshape(-1, )
    runs = np.unique(part_vec)

    indivtrain_em = em.MixVMF(K=m1.emissions[0].K, P=m1.emissions[0].P,
                              X=matrix.indicator(cond_vec), part_vec=part_vec,
                              uniform_kappa=m1.emissions[0].uniform_kappa)
    indivtrain_em.initialize(idata)
    m1.emissions = [indivtrain_em]
    m1.initialize()
    m1, ll, theta, U_indiv = m1.fit_em(iter=200, tol=0.1, fit_emission=True,
                                       fit_arrangement=False,
                                       first_evidence=False)

    Uhat_em_all = []
    Uhat_complete_all = []
    # Loop over to accumulate the runs
    for i in runs:
        ind = part_vec <= i
        m1.emissions[0].X = pt.tensor(matrix.indicator(cond_vec[ind]),
                                      dtype=pt.get_default_dtype())
        m1.emissions[0].part_vec = pt.tensor(part_vec[ind], dtype=pt.int)
        m1.emissions[0].initialize(idata[:, ind, :])

        LL_em = m1.collect_evidence([m1.emissions[0].Estep()])
        Uhat_complete, _ = m1.arrange.Estep(LL_em)
        Uhat_em_all.append(m1.remap_evidence(pt.softmax(LL_em, dim=1)))
        Uhat_complete_all.append(m1.remap_evidence(Uhat_complete))

    Uhat_group = m1.marginal_prob()
    all_eval = [Uhat_group] + Uhat_em_all + Uhat_complete_all

    # Build model for sc2 (testing session):
    #     indivtrain_em = em.MixVMF(K=m1.K,
    m2 = deepcopy(model)
    cond_vec = tinfo['cond_num_uni'].values.reshape(-1, )
    part_vec = tinfo['half'].values.reshape(-1, )
    test_em = em.MixVMF(K=m2.emissions[0].K, P=m2.emissions[0].P,
                        X=matrix.indicator(cond_vec), part_vec=part_vec,
                        uniform_kappa=m2.emissions[0].uniform_kappa)
    test_em.initialize(tdata)
    m2.emissions = [test_em]
    m2.initialize()

    coserr = calc_test_error(m2, tdata, all_eval)

    T = pd.DataFrame()
    for sub in range(coserr.shape[1]):
        for r in range(16):
            D1 = {}
            D1['type'] = ['emissionOnly']
            D1['runs'] = [r + 1]
            D1['coserr'] = [coserr[r + 1, sub]]
            D1['subject'] = [sub + 1]
            T = pd.concat([T, pd.DataFrame(D1)])
            D1 = {}
            D1['type'] = ['emissionAndPrior']
            D1['runs'] = [r + 1]
            D1['coserr'] = [coserr[r + 17, sub]]
            D1['subject'] = [sub + 1]
            T = pd.concat([T, pd.DataFrame(D1)])
        D1 = {}
        D1['type'] = ['group']
        D1['runs'] = [0]
        D1['coserr'] = [coserr[0, sub]]
        D1['subject'] = [sub + 1]
        T = pd.concat([T, pd.DataFrame(D1)])

    return T

def result_1_plot(D):
    gm = D.coserr[D.type == 'group'].mean()
    sb.lineplot(data=D[D.type != 'group'],
                y='coserr', x='runs', hue='type', markers=True, dashes=False)
    plt.xticks(ticks=np.arange(16) + 1)
    plt.axhline(gm, color='b', ls=':')
    # t.ylim([0.21,0.3])
    pass

def result_3_eval(ses1=None, ses2=None):
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
            model_name = [f'Models_03/asym_Ib_space-MNISymC3_K-10_ses-{this_s1}',
                          f'Models_03/asym_Ib_space-MNISymC3_K-10_ses-{this_s2}',
                          f'Models_03/IBC_sessFusion/asym_Ib_space-MNISymC3_K-10_'
                          f'ses-{this_s1}+{this_s2}',
                          f'Models_02/asym_Ib_space-MNISymC3_K-10_ses-{this_s1}',
                          f'Models_02/asym_Ib_space-MNISymC3_K-10_ses-{this_s2}',
                          f'Models_04/IBC_sessFusion/asym_Ib_space-MNISymC3_K-10_'
                          f'ses-{this_s1}+{this_s2}']

            # remove the sessions were used to training to make test sessions
            this_sess = [i for i in sess if i not in [s1, s2]]
            # 1. Run DCBC individual
            res_dcbc = run_dcbc_individual(model_name, 'IBC', this_sess, cond_ind=None,
                                           part_ind=None, indivtrain_ind=None,
                                           indivtrain_values=[0])
            # 2. Run coserr individual
            res_coserr = run_prederror(model_name, 'IBC', this_sess, cond_ind=None,
                                       part_ind=None, eval_types=['group', 'floor'],
                                       indivtrain_ind=None, indivtrain_values=[0])
            # 3. Merge the two dataframe
            res = pd.merge(res_dcbc, res_coserr, how='outer')
            res['sess1_rel'] = reliability[s1]
            res['sess2_rel'] = reliability[s2]
            res['test_sess_out'] = this_s1 + '+' + this_s2
            results = pd.concat([results, res], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    if (ses1 is not None) and (ses2 is not None):
        fname = f'/eval_all_asym_Ib_K-10_{ses1}+{ses2}_on_leftSess.tsv'
    else:
        fname = f'/eval_all_asym_Ib_K-10_twoSess_on_leftSess.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

def result_3_plot(fname):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    # sess_1 = DataSetIBC(base_dir + '/IBC').sessions
    # sess_2 = DataSetIBC(base_dir + '/IBC').sessions
    num_subj = DataSetIBC(base_dir + '/IBC').get_participants().shape[0]
    # Calculate session reliability
    rel, sess = reliability_maps(base_dir, 'IBC', subtract_mean=True,
                                 voxel_wise=True)
    reliability = dict(zip(sess, rel.mean(axis=1)))
    mean_rel = np.array(list(reliability.values())).mean()
    dif_rel = np.array(list(reliability.values())).max() - \
              np.array(list(reliability.values())).min()

    T = pd.DataFrame()
    for s1,s2 in combinations(sess, 2):
        # print(f'- Start evaluating {s1} and {s2}.')
        df = D.loc[(D['test_sess_out'] == s1.split('-')[1] + '+' + s2.split('-')[1])]
        # if df['sess1_rel'].mean() > df['sess2_rel'].mean():
        if reliability[s1] > reliability[s2]:
            df.loc[df.D == s1.split('-')[1], 'D'] = 'good'
            df.loc[df.D == s2.split('-')[1], 'D'] = 'bad'
        else:
            df.loc[df.D == s1.split('-')[1], 'D'] = 'bad'
            df.loc[df.D == s2.split('-')[1], 'D'] = 'good'

        df.loc[df.D == s1.split('-')[1] + '+' + s2.split('-')[1], 'D'] = 'Fusion'
        # Pick the two sessions with larger reliability difference
        if abs(reliability[s1] - reliability[s2]) > 0.5 * dif_rel:
            T = pd.concat([T, df], ignore_index=True)

    plt.figure(figsize=(10,10))
    crits = ['dcbc_group','coserr_group','dcbc_indiv','coserr_floor']
    for i, c in enumerate(crits):
        plt.subplot(2, 2, i + 1)
        sb.barplot(data=T, x='D', y=c, order=['good','bad','Fusion'], hue='common_kappa',
                   hue_order=T['common_kappa'].unique(), errorbar="se")
        plt.legend(loc='lower right')
        if 'coserr' in c:
            plt.ylim(0.4, 1)

    plt.suptitle(f'IBC two sessions fusion - averaged trend')
    plt.show()

def result_4_eval(K=10, t_datasets = ['Mdtb','Pontine','Nishimoto'],
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

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        for s in sess:
            print(f'- Start evaluating {s}.')
            this_sess = DataSetIBC(base_dir + '/IBC').sessions
            this_sess.remove(s)
            model_name = [f'Models_02/asym_Ib_space-MNISymC3_K-{K}_{s}',
                          f'Models_03/asym_Ib_space-MNISymC3_K-{K}_{s}']
            # 1. Run DCBC individual
            res_dcbc = run_dcbc_individual(model_name, ds, 'all', cond_ind=None,
                                           part_ind='half', indivtrain_ind='half',
                                           indivtrain_values=[1,2])
            # 2. Run coserr individual
            res_coserr = run_prederror(model_name, ds, 'all', cond_ind=None,
                                       part_ind='half', eval_types=['group', 'floor'],
                                       indivtrain_ind='half', indivtrain_values=[1,2])
            # 3. Merge the two dataframe
            res = pd.merge(res_dcbc, res_coserr, how='outer')
            results = pd.concat([results, res], ignore_index=True)

        # Additionally, evaluate all IBC sessions fusion on other datasets
        fusion_name = [f'Models_02/asym_Ib_space-MNISymC3_K-{K}',
                       f'Models_03/asym_Ib_space-MNISymC3_K-{K}']
        # 1. Run DCBC individual
        res_dcbc = run_dcbc_individual(fusion_name, ds, 'all', cond_ind=None,
                                       part_ind='half', indivtrain_ind='half',
                                       indivtrain_values=[1, 2])
        # 2. Run coserr individual
        res_coserr = run_prederror(fusion_name, ds, 'all', cond_ind=None,
                                   part_ind='half', eval_types=['group', 'floor'],
                                   indivtrain_ind='half', indivtrain_values=[1, 2])
        # 3. Merge the two dataframe
        res = pd.merge(res_dcbc, res_coserr, how='outer')
        results = pd.concat([results, res], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    if test_ses is not None:
        fname = f'/eval_all_asym_Ib_K-{K}_{test_ses}_on_leftSess.tsv'
    else:
        fname = f'/eval_all_asym_Ib_K-{K}_indivSess_on_leftSess.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

def result_4_plot(fname, common_kappa=True):
    D = pd.read_csv(model_dir + fname, delimiter='\t')

    df = D.loc[(D['common_kappa'] == common_kappa)]

    plt.figure(figsize=(15,10))
    crits = ['dcbc_group','dcbc_indiv','coserr_group','coserr_floor']
    for i, c in enumerate(crits):
        plt.subplot(4, 1, i + 1)
        sb.barplot(data=df, x='test_data', y=c, hue='D', errorbar="se")
        plt.legend('')
        # if 'coserr' in c:
        #     plt.ylim(0.4, 1)

    plt.suptitle(f'IBC individual sessions vs. all sessions fusion, common_kappa={common_kappa}')
    plt.show()

def result_5_eval(K=10, model_type=None, model_name=None,
                  t_datasets=None):
    """Evaluate group and individual DCBC and coserr of all dataset fusion
       and any dataset training standalone on each of the datasets.
    Args:
        K: the number of parcels
        t_datasets (list): a list of test datasets
    Returns:
        Write in evaluation file
    """
    # Preparing model type, name, and test set is not given
    if model_type is None:
        model_type = ['01','02','03','04','05']

    if model_name is None:
        model_name = ['Md','Po','Ni','Ib','MdPoNiIb']

    if t_datasets is None:
        t_datasets = ['Mdtb', 'Pontine', 'Nishimoto', 'Ibc']

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for t in model_type:
        print(f'- Start evaluating Model_{t} - {model_name}...')
        m_name = [f'Models_{t}/asym_{nam}_space-MNISymC3_K-{K}' for nam in model_name]

        for ds in t_datasets:
            print(f'Testdata: {ds}\n')
            # 1. Run DCBC individual
            res_dcbc = run_dcbc_individual(m_name, ds, 'all', cond_ind=None,
                                           part_ind='half', indivtrain_ind='half',
                                           indivtrain_values=[1,2])
            # 2. Run coserr individual
            res_coserr = run_prederror(m_name, ds, 'all', cond_ind=None,
                                       part_ind='half', eval_types=['group', 'floor'],
                                       indivtrain_ind='half', indivtrain_values=[1,2])
            # 3. Merge the two dataframe
            res = pd.merge(res_dcbc, res_coserr, how='outer')
            results = pd.concat([results, res], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    fname = f'/eval_all_asym_Ib_K-{K}_datasetFusion.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

def result_5_plot(fname, model_type='Models_01'):
    D = pd.read_csv(model_dir + fname, delimiter='\t')

    df = D.loc[(D['model_type'] == model_type)]

    plt.figure(figsize=(15,10))
    crits = ['dcbc_group','dcbc_indiv','coserr_group',
             'coserr_floor','coserr_ind2','coserr_ind3']
    for i, c in enumerate(crits):
        plt.subplot(6, 1, i + 1)
        sb.barplot(data=df, x='test_data', y=c, hue='model_name', errorbar="se")
        plt.legend('')
        # if 'coserr' in c:
        #     plt.ylim(0.4, 1)

    plt.suptitle(f'All datasets fusion, {model_type}')
    plt.show()

def concat_eval(model_type,prefix,outfile):
    D = pd.DataFrame()
    for m in model_type:
        fname = model_dir + f'/Models/{m}.tsv'
        T = pd.read_csv(fname, delimiter='\t')
        D = pd.concat([D,T], ignore_index=True)
    oname = model_dir + f'/Models/Evaluation/eval_{prefix}_{outfile}.tsv'
    D.to_csv(oname,index=False,sep='\t')

    pass


if __name__ == "__main__":
    ############# Result 1: individual vs. group improvement #############
    # D = result_1_eval(model_name='Models_04/asym_Md_space-MNISymC3_K-10')
    # fname = model_dir + '/Models/Evaluation_04/indivgroup_all_Md_K-10.tsv'
    # D.to_csv(fname, sep='\t', index=False)
    # figure_indiv_group(D)
    # plt.show()

    ############# Result 2: Simulation on session fusion #############
    # from generativeMRF.notebooks.simulate_fusion import *
    # for k in [5, 10, 20, 30]:
    #     simulation_3(K_true=k, K=5, width=50, nsub_list=np.array([10, 10]),
    #                  M=np.array([40, 20], dtype=int), num_part=1, sigma2=0.5,
    #                  iter=100)

    ############# Result 3: IBC two sessions fusion #############
    # result_3_eval(ses1=None, ses2=None)
    fname = f'/Models/Evaluation/eval_all_asym_Ib_K-10_twoSess_on_leftSess.tsv'
    result_3_plot(fname)

    ############# Result 4: IBC single sessions vs. all sessions fusion #############
    # result_4_eval(K=10, t_datasets=['Mdtb', 'Pontine', 'Nishimoto'])
    # fname = f'/Models/Evaluation/eval_all_asym_Ib_K-10_indivSess_on_leftSess.tsv'
    # result_4_plot(fname, common_kappa=False)

    ############# Result 5: All datasets fusion vs. single dataset #############
    # result_5_eval(K=10)
    # fname = f'/Models/Evaluation/eval_all_asym_Ib_K-10_datasetFusion.tsv'
    # result_5_plot(fname, model_type='Models_04')

    ## For quick copy
    fnames_group = ['Models_01/asym_Ib_space-MNISymC3_K-10',
                    'Models_02/asym_Ib_space-MNISymC3_K-10',
                    'Models_03/asym_Ib_space-MNISymC3_K-10',
                    'Models_04/asym_Ib_space-MNISymC3_K-10',
                    'Models_05/asym_Ib_space-MNISymC3_K-10']

    pass