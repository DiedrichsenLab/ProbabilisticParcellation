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
import Functional_Fusion.dataset as ds
import Functional_Fusion.matrix as matrix
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.evaluation as ev
import ProbabilisticParcellation.util as ut 
import ProbabilisticParcellation.evaluate as ppev

# pytorch cuda global flag
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)


def get_individ_group_mdtb(model, atlas='MNISymC3', localizer_tasks=None):
    """ Gets individual (data only), group, and integrated Uhats for 1-16 runs of first ses-s1 fro, the MDTB data set
    Args:
        model: model name
        atlas: atlas name
        localizer_tasks: list of localizer tasks to use (default: None, use all tasks)
    Returns:
        Uhat_data_all: list (for each run) of parcellations based on data only 
        Uhat_complete_all: list of parcellations for integrated estimate
        Uhat_group: group map 
    """
    idata,iinfo,ids = ds.get_dataset(ut.base_dir,'Mdtb', atlas=atlas,
                                  sess=['ses-s1'], type='CondRun')

    # convert tdata to tensor
    idata = pt.tensor(idata, dtype=pt.get_default_dtype())

    # For backwards compatibility: Deal with old model structures that don't have tmp_list yet
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
    if localizer_tasks is not None:
        # get data, particion vector and condition vector for localizer tasks only
        localizer_ind = np.isin(iinfo['cond_num_uni'], localizer_tasks)
        idata = idata[:, localizer_ind, :]
        cond_vec = cond_vec[localizer_ind]
        part_vec = part_vec[localizer_ind]
        
    runs = np.unique(part_vec)

    # 
    indivtrain_em = em.MixVMF(K=m1.emissions[0].K,
                               P=m1.emissions[0].P,
                               X=matrix.indicator(cond_vec),
                               part_vec=part_vec,
                               uniform_kappa=True)
    indivtrain_em.initialize(idata)
    m1.emissions = [indivtrain_em]
    m1.initialize()
    # Refit the model - while the V vectors should be stable 
    # But kappa needs to be adjusted to single Run data
    m1,ll,theta,U_indiv = m1.fit_em(
                     iter=200, tol=0.1,
                     fit_emission=True,
                     fit_arrangement=False,
                    first_evidence=False)

    Uhat_data_all = []  # Parcellation based only on data 
    Uhat_complete_all = [] # Parcellation based on data and model
    for i in runs:
        ind = part_vec<=i
        m1.emissions[0].X = pt.tensor(matrix.indicator(cond_vec[ind]), dtype=pt.get_default_dtype())
        m1.emissions[0].part_vec = pt.tensor(part_vec[ind], dtype=pt.int)
        m1.emissions[0].initialize(idata[:,ind,:])

        # Estep without using the group map
        LL_em = m1.collect_evidence([m1.emissions[0].Estep()])
        LL_comp = m1.arrange.map_to_arrange(LL_em)
        Uhat_datac = pt.softmax(LL_comp,dim=1)
        Uhat_data = m1.arrange.map_to_full(Uhat_datac)
        Uhat_data = Uhat_data / Uhat_data.sum(dim=1, keepdim=True)
        Uhat_data_all.append(Uhat_data)

        # Normal E-step
        Uhat_complete, _ = m1.arrange.Estep(LL_em)
        Uhat_complete_all.append(Uhat_complete)

    Uhat_group = m1.marginal_prob()
    return Uhat_data_all, Uhat_complete_all, Uhat_group
    

def evaluate_dcbc(Uhat_data,Uhat_complete,Uhat_group,atlas='MNISymC3',max_dist=40, mask=None):
    """Do DCBC evaluation on all and collect in data frame. 
    
    """     
    tds = ds.get_dataset_class(ut.base_dir,dataset='mdtb')
    T = tds.get_participants()
    ut.report_cuda_memory()
    atlas_obj, _ = am.get_atlas(atlas, atlas_dir=ut.base_dir + '/Atlases')
    parcel_group = pt.argmax(Uhat_group, dim=0) + 1
    parcel_data = [pt.argmax(i, dim=1) + 1 for i in Uhat_data]
    parcel_complete = [pt.argmax(i, dim=1) + 1 for i in Uhat_complete]
    del Uhat_complete,Uhat_group,Uhat_data
    pt.cuda.empty_cache()
    ut.report_cuda_memory()
    
    dist = ut.compute_dist(atlas_obj.world.T, resolution=1)
    if mask is not None:
        dist = dist[mask,:]
        dist = dist[:,mask]
        parcel_data = [i[:, mask] for i in parcel_data]
        parcel_complete = [i[:, mask] for i in parcel_complete]
    dist[dist>max_dist]=0
    dist = dist.to_sparse()

    T = pd.DataFrame()
    for s in range(24):    
        print(f'subj {s}')
        tdata,tinfo,tds = ds.get_dataset(ut.base_dir,'Mdtb', atlas=atlas,
                                  sess=['ses-s2'], type='CondHalf',subj=[s])
        tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())
        
        dcbc_group = ppev.calc_test_dcbc(parcel_group, tdata, dist,max_dist=max_dist)
        D1 = {}
        D1['type'] = ['group']
        D1['runs'] = [0]
        D1['dcbc'] = [dcbc_group.item()]
        D1['subject'] = [s + 1]
        T = pd.concat([T, pd.DataFrame(D1)])

        for r in range(len(parcel_data)):
            if mask is not None:
                dcbc_data = ppev.calc_test_dcbc(parcel_data[r][s], tdata[:,:,mask], dist,max_dist=max_dist) 
                dcbc_complete = ppev.calc_test_dcbc(parcel_complete[r][s],tdata[:,:,mask], dist,max_dist=max_dist) 
            else:
                dcbc_data = ppev.calc_test_dcbc(parcel_data[r][s], tdata, dist,max_dist=max_dist) 
                dcbc_complete = ppev.calc_test_dcbc(parcel_complete[r][s],tdata, dist,max_dist=max_dist) 

            D1 = {}
            D1['type'] = ['data']
            D1['runs'] = [r + 1]
            D1['dcbc'] = [dcbc_data.item()]
            D1['subject'] = [s + 1]
            T = pd.concat([T, pd.DataFrame(D1)])
            D1 = {}
            D1['type'] = ['data and group']
            D1['runs'] = [r + 1]
            D1['dcbc'] = [dcbc_complete.item()]
            D1['subject'] = [s + 1]
            T = pd.concat([T, pd.DataFrame(D1)])
        # Group
    return T 


def evaluate_coserr(model,Uhat_data,Uhat_complete,Uhat_group,atlas='MNISymC3'):
    # Do cosine-error evaluation...
    # Build model for sc2 (testing session):
    #     indivtrain_em = em.MixVMF(K=m1.K,
        # Test data set:
    tdata,tinfo,tds = ut.get_dataset(ut.base_dir,'Mdtb', atlas=atlas,
                                  sess=['ses-s2'], type='CondHalf')
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

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

    coserr = ppev.calc_test_error(m2,tdata,[Uhat_group]+Uhat_data+Uhat_complete)

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
    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'
    info,model = ut.load_batch_best(mname,device='cpu')
    Uhat_data,Uhat_complete,Uhat_group = get_individ_group_mdtb(model,atlas='MNISymC3')
    D = evaluate_dcbc(Uhat_data,Uhat_complete,Uhat_group,atlas='MNISymC3',max_dist=70)
    fname = ut.model_dir+ '/Models/Evaluation_03/indivgroup_sym_MdPoNiIbWmDeSo_K68.tsv'
    D.to_csv(fname,sep='\t')
    pass
    # D = pd.read_csv(fname,sep='\t')
    # figure_indiv_group(D)
    # plt.show()
    # pass
