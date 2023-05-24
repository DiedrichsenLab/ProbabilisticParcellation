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
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.evaluation as ev
from ProbabilisticParcellation.util import *
import ProbabilisticParcellation.evaluate as ppev

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


def individ_group(model):
    # Individual training dataset:
    idata,iinfo,ids = get_dataset(base_dir,'Mdtb', atlas='MNISymC3',
                                  sess=['ses-s1'], type='CondRun')

    # Test data set:
    tdata,tinfo,tds = get_dataset(base_dir,'Mdtb', atlas='MNISymC3',
                                  sess=['ses-s2'], type='CondHalf')

    # convert tdata to tensor
    idata = pt.tensor(idata, dtype=pt.get_default_dtype())
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

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
    cond_vec = iinfo['cond_num_uni'].values.reshape(-1,)
    part_vec = iinfo['run'].values.reshape(-1,)
    runs = np.unique(part_vec)

    indivtrain_em = em.wMixVMF(K=m1.emissions[0].K,
                               P=m1.emissions[0].P,
                               X=matrix.indicator(cond_vec),
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
        m1.emissions[0].X = pt.tensor(matrix.indicator(cond_vec[ind]), dtype=pt.get_default_dtype())
        m1.emissions[0].part_vec = pt.tensor(part_vec[ind], dtype=pt.int)
        m1.emissions[0].initialize(idata[:,ind,:])

        LL_em = m1.collect_evidence([m1.emissions[0].Estep()])
        Uhat_complete, _ = m1.arrange.Estep(LL_em)
        Uhat_em_all.append(m1.remap_evidence(pt.softmax(LL_em,dim=1)))
        Uhat_complete_all.append(m1.remap_evidence(Uhat_complete))

    Uhat_group = m1.marginal_prob()
    all_eval = [Uhat_group] + Uhat_em_all + Uhat_complete_all

    # Build model for sc2 (testing session):
    #     indivtrain_em = em.MixVMF(K=m1.K,
    m2 = deepcopy(model)
    cond_vec = tinfo['cond_num_uni'].values.reshape(-1,)
    part_vec = tinfo['half'].values.reshape(-1,)
    test_em = em.MixVMF(K=m2.emissions[0].K,
                            P = m2.emissions[0].P,
                            X = matrix.indicator(cond_vec),
                            part_vec=part_vec,
                            uniform_kappa=True)
    test_em.initialize(tdata)
    m2.emissions = [test_em]
    m2.initialize()

    coserr = ppev.calc_test_error(m2,tdata,all_eval)

    T = pd.DataFrame()
    for sub in range(coserr.shape[1]):
        for r in range(16):
            D1 = {}
            D1['type'] = ['emissionOnly']
            D1['runs'] = [r+1]
            D1['coserr'] = [coserr[r+1,sub]]
            D1['subject'] = [sub+1]
            T = pd.concat([T, pd.DataFrame(D1)])
            D1 = {}
            D1['type'] = ['emissionAndPrior']
            D1['runs'] = [r+1]
            D1['coserr'] = [coserr[r+17,sub]]
            D1['subject'] = [sub+1]
            T = pd.concat([T, pd.DataFrame(D1)])
        D1 = {}
        D1['type'] = ['group']
        D1['runs'] = [0]
        D1['coserr'] = [coserr[0,sub]]
        D1['subject'] = [sub+1]
        T = pd.concat([T, pd.DataFrame(D1)])
        
    return T


def figure_indiv_group(D):
    gm = D.coserr[D.type=='group'].mean()
    sb.lineplot(data=D[D.type!='group'],
                y='coserr',x='runs',hue='type',markers=True, dashes=False)
    plt.xticks(ticks=np.arange(16)+1)
    plt.axhline(gm,color='b',ls=':')
    # t.ylim([0.21,0.3])
    pass

if __name__ == "__main__":
    info,model = load_batch_best('Models_05/asym_Ib_space-MNISymC3_K-10')
    D = individ_group(model)
    fname = model_dir + '/Models/Evaluation_02/indivgroup_prederr_Md_K-10.tsv'
    D.to_csv(fname,sep='\t',index=False)
    pass
    # fname = base_dir+ '/Models/Evaluation_01/indivgroup_prederr_Md_K-20.tsv'
    # D = pd.read_csv(fname,sep='\t')
    # figure_indiv_group(D)
    # plt.show()
    # pass
