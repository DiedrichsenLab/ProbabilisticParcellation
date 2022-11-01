# Script for importing the MDTB data set from super_cerebellum to general format.
import pandas as pd
from pathlib import Path
import numpy as np
import torch as pt
import nibabel as nb
import SUITPy as suit
import matplotlib.pyplot as plt
import seaborn as sb
from copy import copy,deepcopy
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import Functional_Fusion.matrix as matrix
import generativeMRF.full_model as fm
import generativeMRF.spatial as sp
import generativeMRF.arrangements as ar
import generativeMRF.emissions as em
import generativeMRF.evaluation as ev
from ProbabilisticParcellation.util import *

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))


def individ_group(model):
    # Individual training dataset:
    idata,iinfo,ids = get_dataset(base_dir,'Mdtb',
                        atlas='MNISymC3',
                        sess=['ses-s1'],
                        type='CondRun')
    # Test data set:
    # tdata,tinfo,tds = get_dataset(base_dir,'Mdtb',
    #                     atlas='MNISymC3',
    #                     sess=['ses-s2'],
    #                     type='CondHalf')

    # Build the individual training model on session 1:
    m1 = deepcopy(model)
    cond_vec = iinfo['cond_num_uni'].values.reshape(-1,)
    part_vec = iinfo['run'].values.reshape(-1,)
    runs = np.unique(part_vec)

    indivtrain_em = em.MixVMF(K=m1.emissions[0].K,
                            P = m1.emissions[0].P,
                            X = matrix.indicator(cond_vec),
                            part_vec=part_vec,
                            uniform_kappa=True)
    indivtrain_em.initialize(idata)
    m1.emissions = [indivtrain_em]
    m1.initialize()
    m1,ll,theta,U_indiv = m1.fit_em(
                    iter=200, tol=0.1,
                    fit_emission=True,
                    fit_arrangement=False,
                    first_evidence=False)

    Uhat_em_all = []
    Uhat_complete_all = []
    for i in runs:
        ind = part_vec<=i
        m1.emissions[0].X = pt.tensor(matrix.indicator(cond_vec[ind]))
        m1.emissions[0].part_vec = pt.tensor(part_vec[ind], dtype=pt.int)
        m1.emissions[0].initialize(idata[:,ind,:])

        Uhat_em = m1.collect_evidence([m1.emissions[0].Estep()])
        Uhat_complete, _ = m1.arrange.Estep(Uhat_em)
        Uhat_em_all.append(m1.remap_evidence(Uhat_em))
        Uhat_complete_all.append(m1.remap_evidence(Uhat_complete))

    Uhat_group = m1.marginal_prob()
    all_eval = [Uhat_group] + Uhat_em_all + Uhat_complete_all

    # Build model for sc2 (testing session):
    #     indivtrain_em = em.MixVMF(K=m1.K,
    m2 = deepcopy(model)
    cond_vec = tinfo['cond_num_uni'].values.reshape(-1,)
    part_vec = tinfo['half'].values.reshape(-1,)
    test_em = em.MixVMF(K=m2.K,
                            P = m2.emissions[0].P,
                            X = matrix.indicator(cond_vec),
                            part_vec=part_vec,
                            uniform_kappa=True)
    test_em.initialize(tdata)
    m2.emissions = [test_em]
    m2.initialize()
    m2,ll,_,_ = m2.fit_em(
                    iter=200, tol=0.1,
                    fit_emission=True,
                    fit_arrangement=False,
                    first_evidence=False)

    A = ev.calc_test_error(m2,tdata,all_eval)

    for sub, a in enumerate(coserr_Uem):
        D1 = {}
        D1['type'] = ['emissionOnly']
        D1['runs'] = [i]
        D1['coserr'] = [a.item()]
        D1['subject'] = [sub+1]
        T = pd.concat([T, pd.DataFrame(D1)])

    for sub, b in enumerate(coserr_Uall):
        D2 = {}
        D2['type'] = ['emissionAndPrior']
        D2['runs'] = [i]
        D2['coserr'] = [b.item()]
        D2['subject'] = [sub + 1]
        T = pd.concat([T, pd.DataFrame(D2)])

    return T


def figure_indiv_group():
    D = pd.read_csv('scripts/indiv_group_err.csv')
    nf = D['noise_floor'].mean()
    gm = D['group map'].mean()
    T=pd.DataFrame()
    co = ['emission','emisssion+arrangement']
    for i,c in enumerate(['dataOnly_run_','data+prior_run_']):
        for r in range(16):
            dict = {'subj':np.arange(24)+1,
                'cond':[co[i]]*24,
                'run':np.ones((24,))*(r+1),
                'data':(D[f'{c}{r+1:02d}']-D['noise_floor'])+nf}
            T=pd.concat([T,pd.DataFrame(dict)],ignore_index = True)
    fig=plt.figure(figsize=(3.5,5))
    sb.lineplot(data=T,y='data',x='run',hue='cond',markers=True, dashes=False)
    plt.xticks(ticks=np.arange(16)+1)
    plt.axhline(nf,color='k',ls=':')
    plt.axhline(gm,color='b',ls=':')
    plt.ylim([0.21,0.3])
    fig.savefig('indiv_group_err.pdf',format='pdf')
    pass

if __name__ == "__main__":
    info,model = load_batch_best('Models_01/asym_Md_space-MNISymC3_K-20')
    D = individ_group(model)
    # A = pt.load('D:/data/nips_2022_supp/uhat_complete_all.pt')[15]
    # parcel = pt.argmax(A, dim=1) + 1
    # for i in range(parcel.shape[0]):
    #     outname = f'MDTB10_16runs_sub-{i}.nii'
    #     _make_maps(parcel, sub=i, save=True, fname=outname)
    #
    # T, gbase, lb, cos_em, cos_complete, uhat_em_all, uhat_complete_all = learn_runs(K=10, e='VMF',
    #                                                                       runs=np.arange(1, 17))
    # df1 = pt.cat((gbase.reshape(1,-1),lb.reshape(1,-1)), dim=0)
    # df1 = pd.DataFrame(df1).to_csv('coserrs_gb_lb_VMF.csv')
    # T.to_csv('coserrs_VMF.csv')
    #
    # figure_indiv_group()
    pass
