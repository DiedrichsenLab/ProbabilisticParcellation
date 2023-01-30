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
    from depreciated.learn_mdtb import get_mdtb_parcel
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
                         saveFile=f'Evaluation/eval_Ib_{sess1}+{sess2}_dcbc_group_on_leftsessions',
                         device='cuda')
    sb.barplot(x='model_type', y='dcbc', data=res)
    plt.show()


def result_1_eval(model_type='Models_03',model_name='asym_Md_space-MNISymC3_K-10',
                  oname=None):
    info, model = load_batch_best(model_type + f'/{model_name}', device='cuda')
    # Individual training dataset:
    idata, iinfo, ids = get_dataset(base_dir, 'MDTB', atlas='MNISymC3',
                                    sess=['ses-s1'], type='CondRun')

    # Test data set:
    tdata, tinfo, tds = get_dataset(base_dir, 'MDTB', atlas='MNISymC3',
                                    sess=['ses-s2'], type='CondHalf')

    # convert tdata to tensor
    idata = pt.tensor(idata, dtype=pt.get_default_dtype())
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

    # Compatible with old model fitting
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

    # Align with the MDTB parcels
    m_mdtb_align = deepcopy(model)
    MDTBparcel, _ = get_parcel('MNISymC3', parcel_name='MDTB10')
    logpi = ar.expand_mn(MDTBparcel.reshape(1, -1) - 1, m_mdtb_align.arrange.K)
    logpi = logpi.squeeze() * 3.0
    logpi[:, MDTBparcel == 0] = 0
    m_mdtb_align.arrange.logpi = logpi.softmax(dim=0)

    # Build the individual training model on session 1:
    m1 = deepcopy(model)
    MM = [m_mdtb_align, m1]
    Prop = ev.align_models(MM, in_place=True)

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
    all_eval = [Uhat_group] + Uhat_em_all + Uhat_complete_all + ['floor']

    # Build model for sc2 (testing session):
    #     indivtrain_em = em.MixVMF(K=m1.K,
    m2 = deepcopy(model)
    MM = [m1, m2]
    Prop = ev.align_models(MM, in_place=True)

    cond_vec = tinfo['cond_num_uni'].values.reshape(-1, )
    part_vec = tinfo['half'].values.reshape(-1, )
    test_em = em.MixVMF(K=m2.emissions[0].K, P=m2.emissions[0].P,
                        X=matrix.indicator(cond_vec), part_vec=part_vec,
                        uniform_kappa=m2.emissions[0].uniform_kappa)
    test_em.initialize(tdata)
    m2.emissions = [test_em]
    m2.initialize()

    coserr = calc_test_error(m2, tdata, all_eval)

    # Include DCBC evluation in group vs. indiv
    atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')
    dist = compute_dist(atlas.world.T, resolution=1)
    dcbc_group = calc_test_dcbc(pt.argmax(Uhat_group, dim=0) + 1, tdata, dist)
    dcbc_em = [calc_test_dcbc(pt.argmax(i, dim=1) + 1, tdata, dist) for i in Uhat_em_all]
    dcbc_complete = [calc_test_dcbc(pt.argmax(i, dim=1) + 1, tdata, dist) for i in Uhat_complete_all]

    T = pd.DataFrame()
    for sub in range(coserr.shape[1]):
        for r in range(16):
            D1 = {}
            D1['type'] = ['dataOnly']
            D1['runs'] = [r + 1]
            D1['coserr'] = [coserr[r + 1, sub]]
            D1['dcbc'] = [dcbc_em[r][sub].item()]
            D1['subject'] = [sub + 1]
            T = pd.concat([T, pd.DataFrame(D1)])
            D1 = {}
            D1['type'] = ['dataAndPrior']
            D1['runs'] = [r + 1]
            D1['coserr'] = [coserr[r + 17, sub]]
            D1['dcbc'] = [dcbc_complete[r][sub].item()]
            D1['subject'] = [sub + 1]
            T = pd.concat([T, pd.DataFrame(D1)])
        # Group
        D1 = {}
        D1['type'] = ['group']
        D1['runs'] = [0]
        D1['coserr'] = [coserr[0, sub]]
        D1['dcbc'] = [dcbc_group[sub].item()]
        D1['subject'] = [sub + 1]
        T = pd.concat([T, pd.DataFrame(D1)])
        # noise floor
        D1 = {}
        D1['type'] = ['floor']
        D1['runs'] = [0]
        D1['coserr'] = [coserr[-1, sub]]
        D1['dcbc'] = ""
        D1['subject'] = [sub + 1]
        T = pd.concat([T, pd.DataFrame(D1)])

    if oname is not None:
        T.to_csv(fname, sep='\t', index=False)

    return T, [Uhat_group, Uhat_em_all, Uhat_complete_all]

def result_1_plot_curve(fname, oname=None, save=False):
    D = pd.read_csv(fname, delimiter='\t')
    crits = ['coserr', 'dcbc']
    plt.figure(figsize=(10, 5))
    for i, c in enumerate(crits):
        plt.subplot(1, 2, i + 1)
        gm = D[c][D.type == 'group'].mean()
        sb.lineplot(data=D.loc[(D.type != 'group')&(D.type != 'floor')],
                    y=c, x='runs', hue='type', markers=True, dashes=False)
        plt.xticks(ticks=np.arange(16) + 1)
        plt.axhline(gm, color='k', ls=':')
        if c == 'coserr':
            fl = D.coserr[D.type == 'floor'].mean()
            plt.axhline(fl, color='r', ls=':')

    plt.suptitle(f'Individual vs. group, {oname}')
    plt.tight_layout()
    if (oname is not None) and save:
        plt.savefig(res_dir + f'/1.indiv_vs_group/{oname}', format='pdf')

    plt.show()

def result_1_plot_flatmap(Us, sub=0, save=False):
    group, U_em, U_comp = Us[0], Us[1], Us[2]

    oneRun_em = pt.argmax(U_em[0][sub], dim=0) + 1
    allRun_em = pt.argmax(U_em[-1][sub], dim=0) + 1
    oneRun_comp = pt.argmax(U_comp[0][sub], dim=0) + 1
    group = pt.argmax(group, dim=0) + 1
    maps = pt.stack([oneRun_em, allRun_em, oneRun_comp, group])

    # Read and align to the MDTB colors
    MDTBparcel, MDTBcolors = get_parcel('MNISymC3', parcel_name='MDTB10')
    color_file = atlas_dir + '/tpl-SUIT/atl-MDTB10.lut'
    color_info = pd.read_csv(color_file, sep=' ', header=None)
    MDTBcolors = np.zeros((11, 3))
    MDTBcolors[1:11,:]  = color_info.iloc[:,1:4].to_numpy()

    plt.figure(figsize=(25, 25))
    plot_multi_flat(maps.cpu().numpy(), 'MNISymC3', grid=(2, 2), cmap=MDTBcolors,
                    titles=['One run data only',
                            '16 runs data only',
                            'One run data + group probability map',
                            'group probability map'])
    if save:
        plt.suptitle(f'Individual vs. group, MDTB - sub {sub}')
        plt.savefig(res_dir + f'/1.indiv_vs_group/indiv_group_plot_sub_{sub}.png', format='png')

    plt.show()

def result_3_eval(K=10, ses1=None, ses2=None):
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
            model_name = [f'Models_03/asym_Ib_space-MNISymC3_K-{K}_ses-{this_s1}',
                          f'Models_03/asym_Ib_space-MNISymC3_K-{K}_ses-{this_s2}',
                          f'Models_03/IBC_sessFusion/asym_Ib_space-MNISymC3_K-{K}_'
                          f'ses-{this_s1}+{this_s2}',
                          f'Models_04/asym_Ib_space-MNISymC3_K-{K}_ses-{this_s1}',
                          f'Models_04/asym_Ib_space-MNISymC3_K-{K}_ses-{this_s2}',
                          f'Models_04/IBC_sessFusion/asym_Ib_space-MNISymC3_K-{K}_'
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

def result_3_plot(fname, train_model='IBC', ck=None):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    D['relevant'] = ""
    if ck is not None:
        D = D.loc[D['common_kappa']==ck]

    # Calculate session reliability
    rel, sess = reliability_maps(base_dir, train_model, subtract_mean=False,
                                 voxel_wise=True)
    rel_ses, _ = reliability_maps(base_dir, train_model, subtract_mean=False,
                                  voxel_wise=False)
    ses_cor = np.corrcoef(rel)
    # np.fill_diagonal(ses_cor, 0)
    # rel.sort(axis=1)
    # reliability = dict(zip(sess, rel[:,-int(rel.shape[1] * 0.5):].mean(axis=1)))
    reliability = dict(zip(sess, rel_ses[:, 0]))

    T = pd.DataFrame()
    for s1,s2 in combinations(sess, 2):
        # print(f'- Start evaluating {s1} and {s2}.')
        df = D.loc[(D['test_sess_out'] == s1.split('-')[1] + '+' + s2.split('-')[1])]
        # if df['sess1_rel'].mean() > df['sess2_rel'].mean():
        if reliability[s1] >= reliability[s2]:
            df.loc[df.session == s1.split('-')[1], 'session'] = 'sess_1'
            df.loc[df.session == s2.split('-')[1], 'session'] = 'sess_2'
        else:
            df.loc[df.session == s1.split('-')[1], 'session'] = 'sess_2'
            df.loc[df.session == s2.split('-')[1], 'session'] = 'sess_1'

        df.loc[df.session == s1.split('-')[1] + '+' + s2.split('-')[1], 'session'] = 'Fusion'
        # Pick the two sessions with larger reliability difference
        # if abs(reliability[s1] - reliability[s2]) > 0.3 * dif_rel:
        # if (reliability[s1] > mean_rel and reliability[s2] < mean_rel) or \
        #         (reliability[s1] < mean_rel and reliability[s2] > mean_rel):
        # Triage current two sessions into relevant or irrelevant by similarity
        if ses_cor[sess.index(s1)][sess.index(s2)] <= 0:
            print(f'irrelevant sessions: {s1} and {s2}')
            df.loc[df.index, 'relevant'] = False
            T = pd.concat([T, df], ignore_index=True)
        elif ses_cor[sess.index(s1)][sess.index(s2)] >= 0.4:
            print(f'relevant sessions: {s1} and {s2}')
            df.loc[df.index, 'relevant'] = True
            T = pd.concat([T, df], ignore_index=True)
        else:
            pass
            # df.loc[df.index, 'relevant'] = 'Between'
        # T = pd.concat([T, df], ignore_index=True)

    plt.figure(figsize=(22,5))
    crits = ['dcbc_group','dcbc_indiv','coserr_group','coserr_floor']
    for i, c in enumerate(crits):
        plt.subplot(1, 4, i + 1)
        # sb.barplot(data=T, x='session', y=c, order=['good','bad','Fusion'], hue='common_kappa',
        #            hue_order=T['common_kappa'].unique(), errorbar="se")
        sb.lineplot(data=T, x="K", y=c, hue='session', hue_order=['sess_1','sess_2','Fusion'],
                    style='relevant', style_order=T.relevant.unique(), markers=True)
        plt.legend(loc='lower right')

    plt.suptitle(f'IBC two sessions fusion on selected sessions trend, common_kappa={ck}')
    plt.tight_layout()
    plt.savefig('Ibc_twoSessFusion.pdf', format='pdf')
    plt.show()

def result_3_rel_check(fname):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    # Calculate session reliability
    rel, sess = reliability_maps(base_dir, 'IBC', subtract_mean=False,
                                 voxel_wise=False)
    reliability = dict(zip(sess, rel.mean(axis=1)))
    crits = ['dcbc_group', 'dcbc_indiv', 'coserr_group', 'coserr_floor']

    plt.figure(figsize=(25, 5))
    for i, c in enumerate(crits):
        plt.subplot(1, 4, i + 1)
        T = pd.DataFrame()
        for s in sess:
            this_df = D.loc[(D['D'] == s.split('-')[1])].reset_index()
            # T1 = pd.DataFrame({'session': [s], 'reliability': [reliability[s]],
            #                    c: [this_df.loc[(this_df['common_kappa'] == True), c].mean()],
            #                    'common_kappa': [True]})
            # T2 = pd.DataFrame({'session': [s], 'reliability': [reliability[s]],
            #                    c: [this_df.loc[(this_df['common_kappa'] == False), c].mean()],
            #                    'common_kappa': [False]})
            # T = pd.concat([T, T1, T2], ignore_index=True)
            this_df['reliability'] = reliability[s]
            this_df['session'] = s
            T = pd.concat([T, this_df], ignore_index=True)
            for ck in [True, False]:
                plt.text(reliability[s], this_df.loc[(this_df['common_kappa'] == ck), c].mean(),
                         s.split('-')[1], fontdict=dict(color='black', alpha=0.5))

        sb.lineplot(data=T, x="reliability", y=c, hue="common_kappa",
                    hue_order=T['common_kappa'].unique(), errorbar="se",
                    err_style="bars", markers=True, markersize=10)
        # sb.scatterplot(data=T, x="reliability", y=c, hue="common_kappa",
        #                hue_order=T['common_kappa'].unique())

    plt.suptitle(f'IBC individual sessions perfermance vs reliability, test_data=IBC_leftoutSess')
    plt.show()

def result_4_eval(K=[10], t_datasets = ['MDTB','Pontine','Nishimoto'],
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
    model_name += [f'Models_03/asym_Ib_space-MNISymC3_K-{this_k}_{s}'
                   for this_k in K for s in sess]
    model_name += [f'Models_04/asym_Ib_space-MNISymC3_K-{this_k}_{s}'
                   for this_k in K for s in sess]

    # Additionally, add all IBC sessions fusion to list
    model_name += [f'Models_03/asym_Ib_space-MNISymC3_K-{this_k}'
                   for this_k in K]
    model_name += [f'Models_04/asym_Ib_space-MNISymC3_K-{this_k}'
                   for this_k in K]

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        # 1. Run DCBC individual
        res_dcbc = run_dcbc_individual(model_name, ds, 'all', cond_ind=None,
                                       part_ind='half', indivtrain_ind='half',
                                       indivtrain_values=[1,2], device='cuda')
        # 2. Run coserr individual
        res_coserr = run_prederror(model_name, ds, 'all', cond_ind=None,
                                   part_ind='half', eval_types=['group', 'floor'],
                                   indivtrain_ind='half', indivtrain_values=[1,2],
                                   device='cuda')
        # 3. Merge the two dataframe
        res = pd.merge(res_dcbc, res_coserr, how='outer')
        results = pd.concat([results, res], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    if test_ses is not None:
        fname = f'/eval_all_asym_Ib_K-{K}_{test_ses}_on_otherDatasets.tsv'
    else:
        fname = f'/eval_all_asym_Ib_K-10_to_68_indivSess_on_otherDatasets.tsv'
    results.to_csv(wdir + fname, index=False, sep='\t')

def result_4_plot(fname, test_data=None, orderby=None):
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
            order = D.loc[(D['common_kappa'] == orderby)].groupby('D')[c].mean().sort_values().keys(
            ).to_list()
        else:
            order = D.groupby('D')[c].mean().sort_values().keys().to_list()
        sb.barplot(data=D, x='D', y=c, order=order, hue='common_kappa',
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

def result_5_eval(K=10, symmetric='asym', model_type=None, model_name=None,
                  t_datasets=None, return_df=False, k_merged=None, load_best=True):
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
        model_name = ['Md','Po','Ni','Ib','Wm','De','So','MdPoNiIbWmDeSo']

    if t_datasets is None:
        t_datasets = ['MDTB','Pontine','Nishimoto','IBC',
                      'WMFS','Demand','Somatotopic']

    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    m_name = []
    for t in model_type:
        print(f'- Start evaluating Model_{t} - {model_name}...')
        m_name += [f'Models_{t}/{symmetric}_{nam}_space-MNISymC3_K-{K}' for nam in model_name]
        if k_merged is not None:
            m_name = [f'Models_{t}/{symmetric}_{nam}_space-MNISymC3_K-{K}_merged_K-{k_merged}' for nam in model_name]
            load_best=False

    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        # 1. Run DCBC individual
        res_dcbc = run_dcbc_individual(m_name, ds, 'all', cond_ind=None,
                                       part_ind='half', indivtrain_ind='half',
                                       indivtrain_values=[1,2], device='cuda', load_best=load_best)
        # 2. Run coserr individual
        res_coserr = run_prederror(m_name, ds, 'all', cond_ind=None,
                                   part_ind='half', eval_types=['group', 'floor'],
                                   indivtrain_ind='half', indivtrain_values=[1,2],
                                   device='cuda', load_best=load_best)
        # 3. Merge the two dataframe
        res = pd.merge(res_dcbc, res_coserr, how='outer')
        results = pd.concat([results, res], ignore_index=True)

    if return_df:
        return results
    else:
        # Save file
        wdir = model_dir + f'/Models/Evaluation'
        fname = f'/eval_all_{symmetric}_K-{K}_datasetFusion.tsv'
        results.to_csv(wdir + fname, index=False, sep='\t')

def result_5_plot(fname, model_type='Models_01'):
    D = pd.read_csv(model_dir + fname, delimiter='\t')
    df = D.loc[(D['model_type'] == model_type)]
    crits = ['dcbc_group','dcbc_indiv','coserr_group',
             'coserr_floor','coserr_ind2','coserr_ind3']

    plt.figure(figsize=(15, 10))
    for i, c in enumerate(crits):
        plt.subplot(1, 6, i + 1)
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

def eval_arbitrary(model_name=[f'Models_03/asym_Ib_space-MNISymC3_K-10'],
                   t_datasets = ['Mdtb','Pontine','Nishimoto'],
                   test_ses=['all'], fname=None):
    """Evaluate group and individual DCBC and coserr of IBC single
       sessions on all other test datasets.
    Args:
        K: the number of parcels
    Returns:
        Write in evaluation file
    """
    cond_ind = np.array(['cond_num_uni', 'task_num',
                         'reg_id', 'cond_num_uni', 'reg_id'], dtype=object)
    results = pd.DataFrame()
    # Evaluate all single sessions on other datasets
    for ds in t_datasets:
        print(f'Testdata: {ds}\n')
        for s in test_ses:
            print(f'- Start evaluating {s}.')
            # 1. Run DCBC individual
            res_dcbc = run_dcbc_individual(model_name, ds, s, cond_ind=None,
                                           part_ind='half', indivtrain_ind='half',
                                           indivtrain_values=[1,2], device='cuda')
            # 2. Run coserr individual
            res_coserr = run_prederror(model_name, ds, s, cond_ind=None,
                                       part_ind='half', eval_types=['group', 'floor'],
                                       indivtrain_ind='half', indivtrain_values=[1,2],
                                       device='cuda')
            # 3. Merge the two dataframe
            res = pd.merge(res_dcbc, res_coserr, how='outer')
            results = pd.concat([results, res], ignore_index=True)

    # Save file
    wdir = model_dir + f'/Models/Evaluation'
    if fname is not None:
        results.to_csv(wdir + fname, index=False, sep='\t')
    else:
        return results

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
            D = pd.concat([D, res], ignore_index=True)

        D.to_csv(out_name, sep='\t', index=False)


if __name__ == "__main__":
    ############# Making all-in-one #############
    # path = model_dir + '/Models/Evaluation/IBC_twoSessions'
    # oname = model_dir + '/Models/Evaluation/eval_asym_train-Ib_twoSess_test-leftOutSess.tsv'

    path = model_dir + '/Results/2.simulation/eval_all_models'
    oname = model_dir + '/Results/2.simulation/eval_Ktrue_20_Kfit_5to40_all_models.tsv'
    make_all_in_one_tsv(path, out_name=oname)

    ############# Result 1: individual vs. group improvement #############
    # model_type = 'Models_03'
    # model_name = 'asym_Md_space-MNISymC3_K-10'
    # fname = model_dir + f'/Results/1.indiv_vs_group/eval_{model_type}_{model_name}.tsv'
    # D, Us = result_1_eval(model_type=model_type, model_name=model_name, oname=fname)
    #
    # result_1_plot_curve(fname, oname=model_type + f'_{model_name}.pdf')
    # for i in range(24):
    #     result_1_plot_flatmap(Us, sub=i, save=True)

    ############# Result 2: Simulation on session fusion #############
    # from generativeMRF.notebooks.simulate_fusion import *
    # for k in [5, 10, 20, 30]:
    #     simulation_3(K_true=k, K=5, width=50, nsub_list=np.array([10, 10]),
    #                  M=np.array([40, 20], dtype=int), num_part=1, sigma2=0.5,
    #                  iter=100)

    ############# Result 3: IBC two sessions fusion #############
    # for k in [34,40,68]:
    #     result_3_eval(K=k)
    # fname = f'/Models/Evaluation/eval_asym_train-Ib_twoSess_test-leftOutSess.tsv'
    # result_3_rel_check(fname)
    # result_3_plot(fname, ck=True)
    # result_3_plot(f'/Models/Evaluation/eval_all_asym_Md_K-10_indivSess_on_otherDatasets.tsv',
    #               train_model='MDTB')

    ############# Result 4: IBC individual sessions vs. all sessions fusion #############
    result_4_eval(K=[10,17,20,34,40,68], t_datasets=['MDTB', 'Pontine', 'Nishimoto',
                                                     'WMFS', 'Demand', 'Somatotopic'])
    # fname = f'/Models/Evaluation/eval_all_asym_Ib_K-10_indivSess_on_otherDatasets.tsv'
    # result_4_plot(fname, test_data='Pontine', orderby=False)

    ############# Result 5: All datasets fusion vs. single dataset #############
    # T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    # D = pd.DataFrame()
    # for i in range(7):
    #     datasets = [0, 1, 2, 3, 4, 5, 6]
    #     datasets.remove(i)
    #     for k in [10,20,34,40,68]:
    #         datanames = T.two_letter_code[datasets].to_list()
    #         res = result_5_eval(K=k, symmetric='sym', model_type=['03','04'],
    #                             model_name=datanames, t_datasets=[T.name[i]],
    #                             return_df=True)
    #         D = pd.concat([D, res], ignore_index=True)
    # wdir = model_dir + f'/Models/Evaluation/sym'
    # fname = f'/eval_all_sym_MdPoNiIbWmDeSo_K-10_to_68_teston_indivDataset.tsv'
    # D.to_csv(wdir + fname, index=False, sep='\t')
    # # fname = f'/Models/Evaluation/eval_all_asym_MdPoNiIbWmDe_K-10_to_68_teston_So.tsv'
    # # result_5_plot(fname, model_type='Models_03')
    #
    # ############# HCP integration #############
    # D = pd.DataFrame()
    # for k in [10, 20, 34, 40, 68]:
    #     res = result_5_eval(K=k, symmetric='sym', model_type=['03','04'],
    #                         model_name=['MdPoNiIbWmDeSo'],
    #                         return_df=True)
    #     D = pd.concat([D, res], ignore_index=True)
    #
    # wdir = model_dir + f'/Models/Evaluation'
    # fname = f'/eval_all_sym_K-10_to_68_all_teston_leftOneOut.tsv'
    # D.to_csv(wdir + fname, index=False, sep='\t')
    #
    # ############# Check common/separate kappa on different K #############
    # T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    # D = pd.DataFrame()
    # for i in range(7):
    #     datasets = [0, 1, 2, 3, 4, 5, 6]
    #     datasets.remove(i)
    #     for k in [10, 20, 34, 40, 68]:
    #         datanames = ''.join(T.two_letter_code[datasets].tolist())
    #         print(f'----Starting evaluating {datanames}, K={k}, test on {T.name.array[i]}...')
    #         res = result_5_eval(K=k, symmetric='sym', model_type=['03','04'],
    #                             model_name=[datanames], t_datasets=[T.name.array[i]],
    #                             return_df=True)
    #         D = pd.concat([D, res], ignore_index=True)
    #
    # wdir = model_dir + f'/Models/Evaluation'
    # fname = f'/eval_all_sym_K-10_to_68_MdPoNiIbWmDeSo_CV.tsv'
    # D.to_csv(wdir + fname, index=False, sep='\t')
    # fname = f'/Models/Evaluation/eval_all_sym_K-10_to_68_MdPoNiIbWmDeSo_CV.tsv'
    # plot_diffK(fname)
    #
    # ## For quick copy
    # fnames_group = ['Models_01/asym_Ib_space-MNISymC3_K-10',
    #                 'Models_02/asym_Ib_space-MNISymC3_K-10',
    #                 'Models_03/asym_Ib_space-MNISymC3_K-10',
    #                 'Models_04/asym_Ib_space-MNISymC3_K-10',
    #                 'Models_05/asym_Ib_space-MNISymC3_K-10']

    ############# Result 6: Clustered models #############
    T = pd.read_csv(base_dir + '/dataset_description.tsv', sep='\t')
    D = pd.DataFrame()
    datasets = [0, 1, 2, 3, 4, 5, 6]
    datanames = T.two_letter_code[datasets].to_list()

    for i in range(7):
        for k_merged in [10,18,22,26]:
            res = result_5_eval(K=68, symmetric='sym', model_type=['03'],
                                model_name=['MdPoNiIbWmDeSo'], t_datasets=[T.name[i]],
                                return_df=True, k_merged=k_merged)
            D = pd.concat([D, res], ignore_index=True)
    wdir = model_dir + f'/Models/Evaluation/sym'
    fname = f'/eval_all_sym_MdPoNiIbWmDeSo_merged_teston_indivDataset.tsv'
    D.to_csv(wdir + fname, index=False, sep='\t')

    pass