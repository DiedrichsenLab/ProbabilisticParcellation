#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result 1: improving individual maps by adding group prior

Created on 1/5/2023 at 11:06 AM
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
import matplotlib
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

def result_1_plot_flatmap(Us, sub=0, cmap='tab20', save_folder=None):
    group, U_em, U_comp = Us[0], Us[1], Us[2]

    oneRun_em = pt.argmax(U_em[0][sub], dim=0) + 1
    allRun_em = pt.argmax(U_em[-1][sub], dim=0) + 1
    oneRun_comp = pt.argmax(U_comp[0][sub], dim=0) + 1
    group = pt.argmax(group, dim=0) + 1
    maps = pt.stack([oneRun_em, allRun_em, oneRun_comp, group])

    plt.figure(figsize=(25, 25))
    plot_multi_flat(maps.cpu().numpy(), 'MNISymC3', grid=(2, 2), cmap=cmap,
                    titles=['One run data only',
                            '16 runs data only',
                            'One run data + group probability map',
                            'group probability map'])

    plt.suptitle(f'Individual vs. group, MDTB - sub {sub}')
    if save_folder is not None:
        sdir = res_dir + f'/1.indiv_vs_group' + save_folder
        plt.savefig(sdir + f'/indiv_group_plot_sub_{sub}.pdf', format='pdf')

    plt.show()


if __name__ == "__main__":
    for t in ['03']:
        for k in [17]:
            model_type = f'Models_{t}'
            model_name = f'asym_Md_space-MNISymC3_K-{k}'
            fname = model_dir + f'/Results/1.indiv_vs_group/eval_{model_type}_{model_name}.tsv'
            # D, Us = result_1_eval(model_type=model_type, model_name=model_name, oname=fname)

    result_1_plot_curve(fname, oname=model_type + f'_{model_name}.pdf', save=True)

    ########## Making color map ##########
    # cmap = matplotlib.cm.get_cmap('Spectral')
    # colors = cmap(np.linspace(0, 1, 35))
    color_file = atlas_dir + '/tpl-SUIT/atl-MDTB10.lut'
    color_info = pd.read_csv(color_file, sep=' ', header=None)
    colors = np.zeros((11,3))
    colors[1:11,:] = color_info.iloc[:,1:4].to_numpy()
