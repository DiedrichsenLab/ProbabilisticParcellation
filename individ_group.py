# Script for importing the MDTB data set from super_cerebellum to general format.
from time import gmtime
import pandas as pd
from pathlib import Path
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
from scipy.linalg import block_diag
import nibabel as nb
import SUITPy as suit
import generativeMRF.full_model as fm
import generativeMRF.spatial as sp
import generativeMRF.arrangements as ar
import generativeMRF.emissions as em
import generativeMRF.evaluation as ev
import torch as pt
import Functional_Fusion.matrix as matrix
import matplotlib.pyplot as plt
import seaborn as sb
import sys

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))
    

def individ_group(model, e='GME', max_iter=100, run_test=np.arange(58, 122),
               runs=np.arange(1, 17), sub=None, do_plot=True):
    # Individual training dataset: 
    idata,iinfo,ids = get_dataset(base_dir,'Mdtb',
                        atlas='MISymC3',
                        sess='ses-s1',
                        type='CondRun')
    # Test data set: 
    tdata,tinfo,tds = get_dataset(base_dir,'Mdtb',
                        atlas='MISymC3',
                        sess='ses-s2',
                        type='CondHalf')

    # Build the individual training model on session 1: 
    m1 = model.deepcopy()
    cond_vec = iinfo['cond_num_uni'].values.reshape(-1,)
    part_vec = iinfo['run'].values.reshape(-1,)
    runs = np.unique(part_vec)

    indivtrain_em = em.MixVMF(K=m1.K,
                            P = m1.emissions[0].P,
                            X = matrix.indicator(cond_vec),
                            part_vec=part_vec,
                            uniform_kappa=True)
    indivtrain_em.initialize(tdata[:,train_indx,:])
    m1.emissions = [indivtrain_em]
    m1.initialize()
    m1,ll,theta,U_indiv = m1.fit_em(
                    iter=200, tol=0.1,
                    fit_emission=True,
                    fit_arrangement=False,
                    first_evidence=False)

    for i in runs:
        ind = part_vec<=i
        m1.emissions[0].X = matrix.indicator(cond_vec)
        m1.part_vec = pt.tensor(part_vec[ind], dtype=pt.int)
        # Infer on current accumulated runs data
        if e == 'GME':
            U_hat_em = M.emission.Estep(Y=Data_cv[:,acc_run_idx,:])
        elif e == 'VMF':
            U_hat_em = M.emission.Estep(Y=Data_cv[:, acc_run_idx, :], pure_compute=True)
        else:
            raise NameError('Unrecognized emission type.')

        U_hat_complete, _ = M.arrange.Estep(U_hat_em)
        # U_hat_complete, _ = M.Estep(Y=Data_cv[:,acc_run_idx,:])
        uhat_em_all.append(U_hat_em)
        uhat_complete_all.append(U_hat_complete)

    
    all_eval = eval_types + [model.remap_evidence(U_indiv)]


    def calc_test_error(M,tdata,U_hats):

    # train emission model on sc2 by frezzing arrangement model learned from sc1
    data_sc2, Xdesign_sc2, _ = get_mdtb_data(ses_id='ses-s2')
    Xdesign_sc2 = pt.tensor(Xdesign_sc2)
    if e == 'GME':
        em_model2 = em.MixGaussianExp(K=K, N=40, P=P, X=Xdesign_sc2, num_signal_bins=100, std_V=True)
        em_model2.Estep(data_sc2)
    elif e == 'VMF':
        em_model2 = em.MixVMF(K=K, N=40, P=P, X=Xdesign_sc2, uniform_kappa=True)
        em_model2.initialize(data_sc2)
    else:
        raise NameError('Unrecognized emission type.')
    em_model2.Mstep(prior)  # give a good starting value by U_hat learned from sc1
    # M2 = fm.FullModel(M.arrange, em_model2)
    # M2, ll_2, _, U_hat_sc2 = M2.fit_em(Y=data_sc2, iter=max_iter, tol=0.01, fit_arrangement=False)

    group_baseline = ev.coserr(pt.tensor(data_sc2),
                               pt.matmul(Xdesign_sc2, em_model2.V), prior,
                               adjusted=True, soft_assign=True)
    if e == 'GME':
        lower_bound = ev.coserr(pt.tensor(data_sc2),
                                pt.matmul(Xdesign_sc2, em_model2.V),
                                pt.softmax(em_model2.Estep(Y=data_sc2), dim=1),
                                adjusted=True, soft_assign=True)
    elif e == 'VMF':
        lower_bound = ev.coserr(pt.tensor(data_sc2),
                                pt.matmul(Xdesign_sc2, em_model2.V),
                                pt.softmax(em_model2.Estep(Y=data_sc2, pure_compute=True), dim=1),
                                adjusted=True, soft_assign=True)
    else:
        raise NameError('Unrecognized emission type.')

    ############################# Cross-validation starts here #############################
    # Make inference on the selected run's data - run_infer
    Data_cv, Xdesign_cv, D_cv = get_mdtb_data(ses_id='ses-s1', type='CondRun')
    Xdesign_cv = pt.tensor(Xdesign_cv)
    indices, cos_em, cos_complete, uhat_em_all, uhat_complete_all = [],[],[],[],[]
    T = pd.DataFrame()
    for i in runs:
        indices.append(np.asarray(np.where(D_cv.run==i)).reshape(-1))
        acc_run_idx = np.concatenate(indices).ravel()
        M.emission.X = Xdesign_cv[acc_run_idx,:]

        # Infer on current accumulated runs data
        if e == 'GME':
            U_hat_em = M.emission.Estep(Y=Data_cv[:,acc_run_idx,:])
        elif e == 'VMF':
            U_hat_em = M.emission.Estep(Y=Data_cv[:, acc_run_idx, :], pure_compute=True)
        else:
            raise NameError('Unrecognized emission type.')

        U_hat_complete, _ = M.arrange.Estep(U_hat_em)
        # U_hat_complete, _ = M.Estep(Y=Data_cv[:,acc_run_idx,:])
        uhat_em_all.append(U_hat_em)
        uhat_complete_all.append(U_hat_complete)

        # if i == 1:
        #     UEM = pt.argmax(U_hat_em, dim=1)+1
        #     UALL = pt.argmax(U_hat_complete, dim=1) + 1
        #     _plot_maps(UEM, sub=1, color=True, render_type='matplotlib',
        #                save=f'emiloglik_only_{1}_run1.pdf')
        #     _plot_maps(UALL, sub=1, color=True, render_type='matplotlib',
        #                save=f'all_{1}_run1.pdf')

        # if i == 16:
        #     UEM = pt.argmax(U_hat_em, dim=1)+1
        #     UALL = pt.argmax(U_hat_complete, dim=1) + 1
        #     for s in range(24):
        #         _plot_maps(UEM, sub=s, color=True, render_type='matplotlib',
        #                    save=f'emiloglik_only_{s}_run16.png')
                # _plot_maps(UALL, sub=s, color=True, render_type='matplotlib',
                #            save=f'all_{s}_run16.png')

        # Calculate cosine error/u abs error between another test data
        # and U_hat inferred from testing
        coserr_Uem = ev.coserr(pt.tensor(data_sc2),
                               pt.matmul(Xdesign_sc2, em_model2.V),
                               pt.softmax(U_hat_em, dim=1),
                               adjusted=True, soft_assign=True)
        coserr_Uall = ev.coserr(pt.tensor(data_sc2),
                                pt.matmul(Xdesign_sc2, em_model2.V), U_hat_complete,
                                adjusted=True, soft_assign=True)

        cos_em.append(coserr_Uem)
        cos_complete.append(coserr_Uall)
        # uerr_Uem = ev.rmse_YUhat(U_hat_em, pt.tensor(Data[:, 58:90, :]),
        #                          M.emission.V[29:61, :])
        # uerr_Uall = ev.rmse_YUhat(U_hat_complete, pt.tensor(Data[:, 58:90, :]),
        #                          M.emission.V[29:61, :])
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

    return T, group_baseline, lower_bound, cos_em, cos_complete, uhat_em_all, uhat_complete_all


def learn_half(K=10, e='GME', max_iter=100, atlas='SUIT3', run_test=np.arange(58, 122),
                   runs=np.arange(1, 17), sub=None, do_plot=True):

    Data_1, Xdesign_1, partV_1 = get_sess_mdtb(atlas=atlas, ses_id='ses-s1')
    Data_2, Xdesign_2, partV_2 = get_sess_mdtb(atlas=atlas, ses_id='ses-s2')
    Xdesign_1 = pt.tensor(Xdesign_1)
    Xdesign_2 = pt.tensor(Xdesign_2)

    # Make arrangement model and initialize the prior from the MDTB map
    P = Data_1.shape[2]
    prior_w = 7.0  # Weight of prior
    mdtb_parcel, mdtb_colors = get_mdtb_parcel(do_plot=False)
    logpi = ar.expand_mn(mdtb_parcel.reshape(1, P) - 1, K)
    logpi = logpi.squeeze() * prior_w
    # Set parcel 0 to unassigned
    logpi[:, mdtb_parcel == 0] = 0

    # Train on sc 1 data
    ar_model = ar.ArrangeIndependent(K=K, P=P, spatial_specific=True,
                                         remove_redundancy=False)
    if e == 'GME':
        em_model = em.MixGaussianExp(K=K, N=40, P=P, X=Xdesign_1,
                                     num_signal_bins=100, std_V=True)
        em_model.Estep(Data_1)  # sample s and s2 in E-step
    elif e == 'VMF':
        em_model = em.MixVMF(K=K, N=40, P=P, X=Xdesign_1, part_Vec=partV_1, uniform_kappa=True)
        em_model.initialize(Data_1)
    elif e == 'wVMF':
        em_model = em.wMixVMF(K=K, N=40, P=P, X=Xdesign_1, part_vec=partV_1, uniform_kappa=True)
        em_model.initialize(Data_1)
    else:
        raise NameError('Unrecognized emission type.')

    # Initilize parameters from group prior and train the m odel
    mdtb_prior = logpi.softmax(dim=0).unsqueeze(0).repeat(em_model.num_subj,1,1)
    em_model.Mstep(mdtb_prior)
    M = fm.FullModel(ar_model, em_model)
    M, ll, theta, U_hat = M.fit_em(Y=Data_1, iter=max_iter, tol=0.00001, fit_arrangement=True)
    plt.plot(ll, color='b')
    plt.show()

    ### fig a: Plot group prior
    # _plot_maps(pt.argmax(M.arrange.logpi, dim=0) + 1, color=True, render_type='matplotlib',
    #            save='group_prior.pdf')
    prior = pt.softmax(M.arrange.logpi, dim=0).unsqueeze(0).repeat(Data_1.shape[0], 1, 1)
    par_learned = pt.argmax(M.arrange.logpi, dim=0) + 1

    # train emission model on sc2 by frezzing arrangement model learned from sc1
    if e == 'GME':
        em_model2 = em.MixGaussianExp(K=K, N=40, P=P, X=Xdesign_2, num_signal_bins=100, std_V=True)
        em_model2.Estep(Data_2)
    elif e == 'VMF':
        em_model2 = em.MixVMF(K=K, N=40, P=P, X=Xdesign_2, part_Vec=partV_2, uniform_kappa=True)
        em_model2.initialize(Data_2)
    elif e == 'wVMF':
        em_model2 = em.wMixVMF(K=K, N=40, P=P, X=Xdesign_2, part_vec=partV_2, uniform_kappa=True)
        em_model2.initialize(Data_2)
    else:
        raise NameError('Unrecognized emission type.')
    em_model2.Mstep(prior)  # give a good starting value by U_hat learned from sc1
    # M2 = fm.FullModel(M.arrange, em_model2)
    # M2, ll_2, _, U_hat_sc2 = M2.fit_em(Y=data_sc2, iter=max_iter, tol=0.01, fit_arrangement=False)

    group_baseline = ev.coserr(pt.tensor(Data_2),
                               pt.matmul(Xdesign_2, em_model2.V), prior,
                               adjusted=True, soft_assign=True)
    if e == 'GME':
        lower_bound = ev.coserr(pt.tensor(Data_2),
                                pt.matmul(Xdesign_2, em_model2.V),
                                pt.softmax(em_model2.Estep(Y=Data_2), dim=1),
                                adjusted=True, soft_assign=True)
    elif e == 'VMF':
        lower_bound = ev.coserr(pt.tensor(Data_2),
                                pt.matmul(Xdesign_2, em_model2.V),
                                pt.softmax(em_model2.Estep(Y=Data_2), dim=1),
                                adjusted=True, soft_assign=True)
    elif e == 'wVMF':
        lower_bound = ev.coserr(pt.tensor(Data_2),
                                pt.matmul(Xdesign_2, em_model2.V),
                                pt.softmax(em_model2.Estep(Y=Data_2), dim=1),
                                adjusted=True, soft_assign=True)
    else:
        raise NameError('Unrecognized emission type.')

    ############################# Cross-validation starts here #############################
    indices, cos_em, cos_complete, uhat_em_all, uhat_complete_all = [],[],[],[],[]
    T = pd.DataFrame()

    # Infer on current accumulated runs data
    if e == 'GME':
        U_hat_em = M.emission.Estep(Y=Data_1)
    elif e == 'VMF':
        U_hat_em = M.emission.Estep(Y=Data_1)
    elif e == 'wVMF':
        U_hat_em = M.emission.Estep(Y=Data_1)
    else:
        raise NameError('Unrecognized emission type.')

    U_hat_complete, _ = M.arrange.Estep(U_hat_em)
    # U_hat_complete, _ = M.Estep(Y=Data_cv[:,acc_run_idx,:])
    uhat_em_all.append(U_hat_em)
    uhat_complete_all.append(U_hat_complete)

    # Calculate cosine error/u abs error between another test data
    # and U_hat inferred from testing
    coserr_Uem = ev.coserr(pt.tensor(Data_2),
                           pt.matmul(Xdesign_2, em_model2.V),
                           pt.softmax(U_hat_em, dim=1),
                           adjusted=True, soft_assign=True)
    coserr_Uall = ev.coserr(pt.tensor(Data_2),
                            pt.matmul(Xdesign_2, em_model2.V), U_hat_complete,
                            adjusted=True, soft_assign=True)

    cos_em.append(coserr_Uem)
    cos_complete.append(coserr_Uall)

    for sub, a in enumerate(group_baseline):
        D1 = {}
        D1['type'] = ['group']
        D1['coserr'] = [a.item()]
        D1['subject'] = [sub+1]
        T = pd.concat([T, pd.DataFrame(D1)])

    for sub, a in enumerate(lower_bound):
        D1 = {}
        D1['type'] = ['lowerbound']
        D1['coserr'] = [a.item()]
        D1['subject'] = [sub+1]
        T = pd.concat([T, pd.DataFrame(D1)])

    for sub, a in enumerate(coserr_Uem):
        D1 = {}
        D1['type'] = ['emissionOnly']
        D1['coserr'] = [a.item()]
        D1['subject'] = [sub+1]
        T = pd.concat([T, pd.DataFrame(D1)])

    for sub, b in enumerate(coserr_Uall):
        D2 = {}
        D2['type'] = ['emissionAndPrior']
        D2['coserr'] = [b.item()]
        D2['subject'] = [sub + 1]
        T = pd.concat([T, pd.DataFrame(D2)])

    return T, group_baseline, lower_bound, par_learned

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


def _plot_vmf_wvmf(T, T2):
    """Plot the evaluation of wVMF and VMF
    Args:
        T: VMF
        T2: wVMF
    Returns:
        plot
    """
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.bar(['emission', 'emission+prior'], [T[T['type'] == 'emissionOnly']['coserr'].mean(),
                                             T[T['type'] == 'emissionAndPrior']['coserr'].mean()],
            yerr=[T[T['type'] == 'emissionOnly']['coserr'].std()/np.sqrt(24),
                  T[T['type'] == 'emissionOnly']['coserr'].std()/np.sqrt(24)])
    plt.axhline(y=T[T['type'] == 'group']['coserr'].mean(), color='r', linestyle=':')
    plt.axhline(y=T[T['type'] == 'lowerbound']['coserr'].mean(), color='k', linestyle=':')
    plt.ylim(0.2, 0.32)
    plt.title('VMF')

    plt.subplot(122)
    plt.bar(['emission', 'emission+prior'], [T2[T2['type'] == 'emissionOnly']['coserr'].mean(),
                                             T2[T2['type'] == 'emissionAndPrior']['coserr'].mean()],
            yerr=[T2[T2['type'] == 'emissionOnly']['coserr'].std()/np.sqrt(24),
                  T2[T2['type'] == 'emissionOnly']['coserr'].std()/np.sqrt(24)])
    plt.axhline(y=T2[T2['type'] == 'group']['coserr'].mean(), color='r', linestyle=':')
    plt.axhline(y=T2[T2['type'] == 'lowerbound']['coserr'].mean(), color='k', linestyle=':')
    plt.ylim(0.2, 0.32)
    plt.title('wVMF')
    plt.show()


if __name__ == "__main__":

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
