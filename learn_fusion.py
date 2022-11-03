# Script for importing the MDTB data set from super_cerebellum to general format.
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import Functional_Fusion.matrix as matrix
from scipy.linalg import block_diag
import nibabel as nb
import SUITPy as suit
import generativeMRF.full_model as fm
import generativeMRF.spatial as sp
import generativeMRF.arrangements as ar
import generativeMRF.emissions as em
import generativeMRF.evaluation as ev
import torch as pt
from learn_mdtb import get_mdtb_parcel
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import pickle
from copy import deepcopy

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))
    

def build_data_list(datasets,
                atlas = 'MNISymC3',
                sess = None,
                cond_ind = None,
                type = None,
                part_ind=None,
                subj = None,
                join_sess = True):
    """Builds list of datasets, cond_vec, part_vec, subj_ind
    from different data sets 
    Args:
        datasets (list): Names of datasets to include
        atlas (str): Atlas indicator
        sess (list): list of 'all' or list of sessions 
        design_ind (list, optional): _description_. Defaults to None.
        part_ind (list, optional): _description_. Defaults to None.
        subj (list, optional): _description_. Defaults to None.
        join_sess (bool, optional): Model the sessions with a single model . Defaults to True.
    Returns: 
        data,
        cond_vec,
        part_vec,
        subj_ind
    """
    n_sets = len(datasets)
    data = []
    cond_vec = []
    part_vec = [] 
    subj_ind = []

    # Set defaults for data sets:
    if sess is None:
        sess = ['all'] * n_sets
    if part_ind is None: 
        part_ind = [None] * n_sets
    if cond_ind is None: 
        cond_ind = [None] * n_sets
    if type is None: 
        type = [None] * n_sets

    sub = 0 
    # Run over datasets get data + design 
    for i in range(n_sets):
        dat,info,ds = get_dataset(base_dir,datasets[i],
                atlas=atlas,
                sess=sess[i],
                type=type[i])
        # Sub-index the subjects: 
        if subj is not None:
            dat = dat[subj[i],:,:]
        n_subj = dat.shape[0]

        # Find correct indices 
        if cond_ind[i] is None:
            cond_ind[i] = ds.cond_ind
        if part_ind[i] is None:
            part_ind[i] = ds.part_ind
        # Make different sessions either the same or different
        if join_sess:
            data.append(dat)
            cond_vec.append(info[cond_ind[i]].values.reshape(-1,))
            part_vec.append(info[part_ind[i]].values.reshape(-1,))
            subj_ind.append(np.arange(sub,sub+n_subj))
        else: 
            for s in ds.sessions:
                indx = info.sess == s
                data.append(dat[:,indx,:])
                cond_vec.append(info[cond_ind[i]].values[indx].reshape(-1,))
                part_vec.append(info[part_ind[i]].values[indx].reshape(-1,))
                subj_ind.append(np.arange(sub,sub+n_subj))
        sub+=n_subj
    return data,cond_vec,part_vec,subj_ind 

def batch_fit(datasets,sess,
                type=None,cond_ind=None,part_ind=None,subj=None,
                atlas=None,
                K=10,
                arrange='independent',
                emission='VMF',
                n_rep=3, n_inits=10, n_iter=80,first_iter=10,
                name=None,
                uniform_kappa = True,
                join_sess = True,
                weighting=None):
    """ Executes a set of fits starting from random starting values
    selects the best one from a batch and saves them 

    Args:
        model_type (str): String indicating model_type
        datasets (list): List of dataset names to be used as training
        sess (list): List of list of sessions to be used for each
        type (list): List the type  
        cond_ind (list): Name of the info-field that indicates the condition
        part_ind (list): Name of the field indicating independent partitions of the data
        subj (list, optional): _description_. Defaults to None
        atlas (Atlas): Atlas to be used. Defaults to None.
        K (int): Number of parcels. Defaults to 10.
        arrange (str): Type of arangement model. Defaults to 'independent'.
        emission (list / strs): Type of emission models. Defaults to 'VMF'.
        n_inits (int): Number of random starting values. default: 10
        n_iter (int): Maximal number of iterations per fit: default: 20
        save (bool): Save the resulting fits? Defaults to True.
        name (str): Name of model (for filename). Defaults to None.

    Returns:
        info (pd.DataFrame):
    """
    data,cond_vec,part_vec,subj_ind = build_data_list(datasets,
                atlas = atlas.name,
                sess = sess,
                cond_ind = cond_ind,
                type = type,
                part_ind=part_ind,
                subj = subj,
                join_sess = join_sess)
    # Load all necessary data and designs
    n_sets = len(data)

    # Build the model 
    # Check for size of Atlas + whether symmetric
    if isinstance(atlas,am.AtlasVolumeSymmetric):
        P_arrange = atlas.Psym
        K_arrange = np.ceil(K/2).astype(int)
    else:
        P_arrange = atlas.P
        K_arrange = K

    # Initialize arrangement model
    if arrange=='independent':
        ar_model = ar.ArrangeIndependent(K=K_arrange, P=P_arrange,
                                            spatial_specific=True,
                                            remove_redundancy=False)
    else:
        raise(NameError(f'unknown arrangement model:{arrange}'))

    # Initialize emission models
    em_models=[]
    for j,ds in enumerate(data):
        if emission=='VMF':
            em_model = em.MixVMF(K=K, 
                                P=atlas.P,
                                X=matrix.indicator(cond_vec[j]), 
                                part_vec=part_vec[j],
                                uniform_kappa=uniform_kappa)
        else:
            raise((NameError(f'unknown emission model:{emission}')))
        em_models.append(em_model)

    # Make a full fusion model
    if isinstance(atlas,am.AtlasVolumeSymmetric):
            M = fm.FullMultiModelSymmetric(ar_model, em_models,
                                            atlas.indx_full,atlas.indx_reduced,
                                            same_parcels=False)
    else:
            M = fm.FullMultiModel(ar_model, em_models)

    # Step 5: Estimate the parameter thetas to fit the new model using EM

    # Somewhat hacky: Weight different datasets differently 
    if weighting is not None: 
        M.ds_weight = weighting # Weighting for each dataset

    # Initialize data frame for results
    models=[]
    n_fits = n_rep
    info = pd.DataFrame({'name':[name]*n_fits,
                            'atlas':[atlas.name]*n_fits,
                            'K':[K]*n_fits,
                            'datasets':[datasets]*n_fits,
                            'sess':[sess]*n_fits,
                            'type':[type]*n_fits,
                            'subj':[subj]*n_fits,
                            'arrange':[arrange]*n_fits,
                            'emission':[emission]*n_fits,
                            'loglik':[np.nan]*n_fits,
                            'weighting': [weighting]*n_fits});


    # Iterate over the number of fits
    ll = np.empty((n_fits,n_iter))
    for i in range(n_fits):
        print(f'fit: {i}')
        # Copy the obejct (without data)
        m = deepcopy(M)
        # Attach the data
        m.initialize(data,subj_ind = subj_ind)

        m, ll, theta, U_hat, ll_init = m.fit_em_ninits(
                                        iter=n_iter,
                                        tol=0.01, 
                                        fit_arrangement=True,
                                        n_inits=n_inits,
                                        first_iter=first_iter)
        info.loglik.at[i] = ll[-1]
        m.clear()
        models.append(m)

    # Align the different models
    models = np.array(models,dtype=object)
    # ev.align_fits(models)

    return info,models

def fit_all(set_ind=[0,1,2,3],K=10,model_type='01',weighting=None):
    # Data sets need to numpy arrays to allow indixing by list
    datasets = np.array(['Mdtb','Pontine','Nishimoto','Ibc', 'Hcp'],
                    dtype = object)
    sess = np.array(['all','all','all','all', 'all'],
            dtype = object)
    type = np.array(['CondHalf','TaskHalf','CondHalf','CondHalf', 'NetRun'],
            dtype = object)

    cond_ind= np.array(['cond_num_uni','task_num',
                          'reg_id','cond_num_uni', 'reg_id'],dtype = object)
    part_ind = np.array(['half','half','half','half', 'half']
        ,dtype = object)

    # Use specific mask / atlas. 
    mask = base_dir + '/Atlases/tpl-MNI152NLIn2000cSymC/tpl-MNISymC_res-3_gmcmask.nii'
    atlas = [am.AtlasVolumetric('MNISymC3',mask_img=mask),
             am.AtlasVolumeSymmetric('MNISymC3',mask_img=mask)]

    # Give a overall name for the type of model
    mname =['asym','sym']

    # Provide different setttings for the different model types 
    if model_type=='01':
        uniform_kappa = True
        join_sess = True
    elif model_type=='02':
        uniform_kappa = False
        join_sess = True
    elif model_type[:6]=='01-HCP':
        uniform_kappa = True
        weighting = np.repeat(1, len(set_ind)-1).tolist()
        hcp_weight = model_type.split('HCP')[1]
        weighting.extend([float(f'{hcp_weight[0]}.{hcp_weight[1]}')])
        join_sess = True
    elif model_type == '03':
        uniform_kappa = True
        join_sess = False
    elif model_type == '04':
        uniform_kappa = False
        join_sess = False

    #Generate a dataname from first two letters of each training data set 
    dataname = [datasets[i][0:2] for i in set_ind]
    
    print(f'Fitting {model_type} with K={K}')
    for i in [0,1]:
        name = mname[i] + '_' + ''.join(dataname) 
        print(f'{name}')
        info,models = batch_fit(datasets[set_ind],
              sess = sess[set_ind],
              type = type[set_ind],
              cond_ind = cond_ind[set_ind],
              part_ind = part_ind[set_ind],
              atlas=atlas[i],
              K=K,
              name=name,
              n_inits=20, 
              n_iter=200,
              n_rep=10,
              first_iter=30,
              join_sess = join_sess,
              uniform_kappa = uniform_kappa,
              weighting=weighting)

        # Save the fits and information
        wdir = base_dir + f'/Models/Models_{model_type}'
        Path(wdir).mkdir(exist_ok=True)
        fname = f'/{name}_space-{atlas[i].name}_K-{K}'
        info.to_csv(wdir + fname + '.tsv',sep='\t')
        with open(wdir + fname + '.pickle','wb') as file:
            pickle.dump(models,file)

if __name__ == "__main__":
    for k in [10,20,34]:
        fit_all([0,1,2,3,4],k,model_type='01-HCP05') # rerun
        fit_all([0,1,2,3,4],k,model_type='01-HCP02')
        fit_all([0,1,2,3,4],k,model_type='01-HCP03')
        fit_all([0,1,2,3,4],k,model_type='01-HCP07')
    
    for k in [10,20,34]:
        fit_all([0,1,2,3,4],k,model_type='01-HCP01')
        fit_all([0,1,2,3,4],k,model_type='01-HCP02')
        fit_all([0,1,2,3,4],k,model_type='01-HCP04')


    pass
