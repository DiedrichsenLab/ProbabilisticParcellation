# Evaluate cerebellar parcellations
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
from ProbabilisticParcellation.util import *
# from DCBC.DCBC_vol import compute_DCBC, compute_dist

# pytorch cuda global flag
# pt.set_default_tensor_type(pt.cuda.FloatTensor
#                            if pt.cuda.is_available() else
#                            pt.FloatTensor)

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

def calc_test_error(M,tdata,U_hats):
    """Evaluates the predictions from a trained full model on some testdata.
    The full model consists of a trained arrangement model and is
    combined with untrained emission model for the test data.
    The function then trains the emission model based on N-1 subjects
    (arrangement fixed) and evaluates the left-out subjects using different
    parcellations (Uhats)

    Args:
        M (full model): Full model including emission model for test data
        tdata (ndarray): (numsubj x N x P) array of test data
        U_hats (list): List of strings or tensors, the indi. parcellation
            'group':   Group-parcellation
            'floor':   Noise-floor (E-step on left-out subject)
            pt.tensor: Any arbitrary Individual parcellation based on outside data
    Returns:
        A num_eval x num_subj matrix of cosine errors
    """
    num_subj = tdata.shape[0]
    subj = np.arange(num_subj)
    group_parc = M.marginal_prob()
    pred_err = np.empty((len(U_hats),num_subj))
    for s in range(num_subj):
        print(f'Subject:{s}')
        # initialize the emssion model using all but one subject
        M.emissions[0].initialize(tdata[subj != s,:,:])
        # For fitting an emission model witout the arrangement model,
        # We can not without multiple starting values
        M.initialize()
        M,ll,theta,Uhat = M.fit_em(
                iter=200, tol=0.1,
                fit_emission=True,
                fit_arrangement=False,
                first_evidence=False)
        X = M.emissions[0].X
        dat = pt.linalg.pinv(X) @ tdata[subj==s,:,:]
        for i,crit in enumerate(U_hats):
            if crit=='group':
                U = group_parc
            elif crit=='floor':
                # U,ll = M.Estep(Y=pt.tensor(tdata[subj==s,:,:]).unsqueeze(0))
                M.emissions[0].initialize(tdata[subj == s,:,:])
                U = pt.softmax(M.emissions[0].Estep(tdata[subj == s, :, :]), dim=1)
                U = M.remap_evidence(U)
            elif crit.ndim == 2:
                U = crit
            elif crit.ndim == 3: 
                U = crit[subj==s,:,:]
            else: 
                raise(NameError("U_hats needs to be 'group','floor',a 2-d or 3d-tensor"))
            a=ev.coserr(dat, M.emissions[0].V, U,
                        adjusted=True, soft_assign=True)
            pred_err[i,s] = a
    return pred_err


def calc_test_dcbc(parcels, testdata, dist, trim_nan=False):
    """DCBC: evaluate the resultant parcellation using DCBC
    Args:
        parcels (np.ndarray): the input parcellation:
            either group parcellation (1-dimensional: P)
            individual parcellation (num_subj x P )
        dist (<AtlasVolumetric>): the class object of atlas
        testdata (np.ndarray): the functional test dataset,
                                shape (num_sub, N, P)
        trim_nan (boolean): if true, make the nan voxel label will be
                            removed from DCBC calculation. Otherwise,
                            we treat nan voxels are in the same parcel
                            which is label 0 by default.
    Returns:
        dcbc_values (np.ndarray): the DCBC values of subjects
    """

    #
    # if trim_nan:  # mask the nan voxel pairs distance to nan
    #     dist[np.where(np.isnan(parcels))[0], :] = np.nan
    #     dist[:, np.where(np.isnan(parcels))[0]] = np.nan

    dcbc_values = []
    for sub in range(testdata.shape[0]):
        print(f'Subject {sub}',end=':')
        tic = time.perf_counter()
        if parcels.ndim==1:
            D = compute_DCBC(maxDist=110, binWidth=5, parcellation=parcels,
                             dist=dist, func=testdata[sub].T)
        else:
            D = compute_DCBC(maxDist=110, binWidth=5, parcellation=parcels[sub],
                             dist=dist, func=testdata[sub].T)
        dcbc_values.append(D['DCBC'])
        toc = time.perf_counter()
        print(f"{toc-tic:0.4f}s")
    return pt.stack(dcbc_values)


def run_prederror(model_names,test_data,test_sess,
                    cond_ind,part_ind=None,
                    eval_types=['group','floor'],
                    indivtrain_ind=None,indivtrain_values=[0]):
    """ Calculates a prediction error using a test_data set
    and test_sess.
    if indivtrain_ind is given, it splits the test_data set
    again and uses one half to derive an individual parcellation
    (using the model) and the other half to evaluate it.
    The Means of the parcels are always estimated on N-1 subjects
    and evaluated on the Nth left-out subject

    Args:
        model_names (list or str): Name of model fit (tsv/pickle file)
        test_data (str): Name of test data set
        test_sess (list): List or sessions to include into test_data
        cond_ind (str): Fieldname of the condition vector in test-data info
        part_ind (str): Fieldname of partition vector in test-data info
        eval_types (list): Defaults to ['group','floor'].
        indivtrain_ind (str): If given, data will be split for individual
             training along this field in test-data info. Defaults to None.
        indivtrain_values (list): Values of field above to be taken as
             individual training sets.

    Returns:
        data-frame with model evalution
    """
    tdata,tinfo,tds = get_dataset(base_dir,test_data,
                              atlas='MNISymC3',sess=test_sess)
    # convert tdata to tensor
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

    # For testing: tdata=tdata[0:5,:,:]
    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    if not isinstance(model_names,list):
        model_names = [model_names]

    # Get condition and partition vector of test data
    if cond_ind is None:
        cond_ind = tds.cond_ind
    cond_vec = tinfo[cond_ind].values.reshape(-1,)
    if part_ind is None:
        part_vec = np.zeros((tinfo.shape[0],),dtype=int)
    else:
        part_vec = tinfo[part_ind].values

    # Decide how many splits we need
    if indivtrain_ind is None:
        n_splits = 1
    else:
        n_splits = len(indivtrain_values)

    # Now loop over possible models we want to evaluate
    for i, model_name in enumerate(model_names):
        print(f"Doing model {model_name}\n")
        minfo, model = load_batch_best(f"{model_name}")
        model_kp = model.emissions[0].uniform_kappa
        this_res = pd.DataFrame()
        # Loop over the splits - if split then train a individual model
        for n in range(n_splits):
            # ------------------------------------------
            # Train an emission model on the individual training data
            # and get a Uhat (individual parcellation) from it.
            if indivtrain_ind is not None:
                train_indx = tinfo[indivtrain_ind]==indivtrain_values[n]
                test_indx = tinfo[indivtrain_ind]!=indivtrain_values[n]
                indivtrain_em = em.MixVMF(K=minfo.K, N=40,
                            P = model.emissions[0].P,
                            X = matrix.indicator(cond_vec[train_indx]),
                            part_vec=part_vec[train_indx],
                            uniform_kappa=model_kp)
                indivtrain_em.initialize(tdata[:,train_indx,:])
                model.emissions = [indivtrain_em]
                model.initialize()
                m,ll,theta,U_indiv = model.fit_em(iter=200, tol=0.1,
                                                  fit_emission=True,
                                                  fit_arrangement=False,
                                                  first_evidence=False)
                # Add individual U_hat data (emission) only
                Uhat_em = pt.softmax(m.emissions[0].Estep(), dim=1)
                # Uhat_compl, _ = m.arrange.Estep(m.emissions[0].Estep())
                all_eval = eval_types + [model.remap_evidence(Uhat_em)] + \
                           [model.remap_evidence(U_indiv)]
            else:
                test_indx =  np.ones((tinfo.shape[0],),dtype=bool)
                all_eval = eval_types
            # ------------------------------------------
            # Now build the model for the test data and crossvalidate
            # across subjects
            em_model = em.MixVMF(K=minfo.K, N=40,
                            P=model.emissions[0].P,
                            X=matrix.indicator(cond_vec[test_indx]),
                            part_vec=part_vec[test_indx],
                            uniform_kappa=model_kp)
            # Add this single emission model
            model.emissions = [em_model]
            # recalculate total number parameters
            model.nparams = model.arrange.nparams + em_model.nparams
            # Calculate cosine error
            res = calc_test_error(model,tdata[:,test_indx,:],all_eval)
            # ------------------------------------------
            # Collect the information from the evaluation
            # in a data frame
            ev_df = pd.DataFrame({'model_name':[minfo['name']]*num_subj,
                            'atlas':[minfo.atlas]*num_subj,
                            'K':[minfo.K]*num_subj,
                            'train_data':[minfo.datasets]*num_subj,
                            'train_loglik':[minfo.loglik]*num_subj,
                            'test_data':[test_data]*num_subj,
                            'indivtrain_ind':[indivtrain_ind]*num_subj,
                            'indivtrain_val':[indivtrain_values[n]]*num_subj,
                            'subj_num':np.arange(num_subj),
                            'common_kappa':[model_kp]*num_subj})
            # Add all the evaluations to the data frame
            for e,ev in enumerate(all_eval):
                if isinstance(ev,str):
                    ev_df['coserr_' + ev]=res[e,:]
                else:
                    ev_df[f'coserr_ind{e}']=res[e,:]
            this_res = pd.concat([this_res, ev_df], ignore_index=True)

        # Concate model type
        this_res['model_type'] = i
        results = pd.concat([results, this_res], ignore_index=True)

    return results


def run_dcbc_group(par_names,space,test_data,test_sess='all',saveFile=None):
    """ Run DCBC group evaluation

    Args:
        par_names (list): List of names for the parcellations to evaluate    
                Can be either 
                    nifti files (*_dseg.nii) or 
                    models (*.npy)
        space (str): Atlas space (SUIT3, MNISym3C)... 
        test_data (str): Data set string 
        test_sess (str, optional): Data set test. Defaults to 'all'.

    Returns:
        DataFrame: Results
    """
    tdata,tinfo,tds = get_dataset(base_dir,test_data,
                              atlas=space,sess=test_sess)
    atlas = am.get_atlas(space,atlas_dir=base_dir + '/Atlases')
    dist = compute_dist(atlas.world.T,resolution=1)

    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    if not isinstance(par_names,list):
        par_names = [par_names]

    # convert tdata to tensor
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())
    # parcel = np.empty((len(model_names), atlas.P))
    results = pd.DataFrame()
    for i, pn in enumerate(par_names):
        fileparts = pn.split('/')
        pname = fileparts[-1]
        pname_parts = pname.split('.')
        print(f'evaluating {pname}')
        if pname_parts[-1]=='pickle':
            minfo, model = load_batch_best(f"{fileparts[-2]}/{pname_parts[-2]}")
            Prop = model.marginal_prob()
            par = pt.argmax(Prop,dim=0)+1
        elif pname_parts[-1]=='nii':
            par = atlas.sample_nifti(pn,0)
        # Initialize result array
        if i == 0:
            dcbc = pt.zeros((len(par_names), tdata.shape[0]))
        print(f"Number zeros {(par==0).sum()}")
        dcbc[i, :] = calc_test_dcbc(par, tdata, dist)
        num_subj = tdata.shape[0]

        ev_df = pd.DataFrame({'fit_type': [fileparts[0]] * num_subj,
                            'model_name': [pname_parts[-2]] * num_subj,
                            'test_data': [test_data] * num_subj,
                            'subj_num': np.arange(num_subj),
                            'dcbc': dcbc[i,:].cpu().numpy()
                            })
        results = pd.concat([results, ev_df], ignore_index=True)

    if saveFile is not None:
        oname = model_dir + f'/Models/{saveFile}.tsv'
        results.to_csv(oname, index=False, sep='\t')

    return results


def run_dcbc_individual(model_names, test_data, test_sess,
                    cond_ind=None,part_ind=None,
                    indivtrain_ind=None,indivtrain_values=[0]):
    """ Calculates DCBC using a test_data set
    and test_sess.
    if indivtrain_ind is given, it splits the test_data set
    again and uses one half to derive an individual parcellation
    (using the model) and the other half to evaluate it.
    The Means of the parcels are always estimated on N-1 subjects
    and evaluated on the Nth left-out subject

    Args:
        model_names (list or str): Name of model fit (tsv/pickle file)
        test_data (str): Name of test data set
        test_sess (list): List or sessions to include into test_data
        cond_ind (str): Fieldname of the condition vector in test-data info
        part_ind (str): Fieldname of partition vector in test-data info
        indivtrain_ind (str): If given, data will be split for individual
             training along this field in test-data info. Defaults to None.
        indivtrain_values (list): Values of field above to be taken as
             individual training sets.

    Returns:
        data-frame with model evalution
    """
    tdata,tinfo,tds = get_dataset(base_dir,test_data,
                              atlas='MNISymC3',sess=test_sess)
    atlas = am.get_atlas('MNISymC3',atlas_dir=base_dir + '/Atlases')
    dist = compute_dist(atlas.world.T,resolution=1)

    # convert tdata to tensor
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())
    # For testing: tdata=tdata[0:5,:,:]
    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    if not isinstance(model_names,list):
        model_names = [model_names]

    # Get condition vector of test data
    if cond_ind is None:
        # get default cond_ind from testdataset
        cond_vec = tinfo[tds.cond_ind].values.reshape(-1,)
    else:
        cond_vec = tinfo[cond_ind].values.reshape(-1,)

    # Get partition vector of test data
    if part_ind is None:
        part_vec = np.zeros((tinfo.shape[0],),dtype=int)
    else:
        part_vec = tinfo[part_ind].values

    # Decide how many splits we need
    if indivtrain_ind is None:
        n_splits = 1
    else:
        n_splits = len(indivtrain_values)

    # Now loop over possible models we want to evaluate
    for i, model_name in enumerate(model_names):
        print(f"Doing model {model_name}\n")
        minfo, model = load_batch_best(f"{model_name}")
        Prop = model.marginal_prob()

        this_res = pd.DataFrame()
        # Loop over the splits - if split then train a individual model
        for n in range(n_splits):
            # ------------------------------------------
            # Train an emission model on the individual training data and get a Uhat (individual parcellation) from it.
            if indivtrain_ind is not None:
                train_indx = tinfo[indivtrain_ind]==indivtrain_values[n]
                test_indx = tinfo[indivtrain_ind]!=indivtrain_values[n]
                indivtrain_em = em.MixVMF(K=minfo.K, N=40,
                            P = model.emissions[0].P,
                            X = matrix.indicator(cond_vec[train_indx]),
                            part_vec=part_vec[train_indx],
                            uniform_kappa=model.emissions[0].uniform_kappa)
                indivtrain_em.initialize(tdata[:,train_indx,:])
                model.emissions = [indivtrain_em]
                model.initialize()
                # Gets us the individual parcellation
                model,ll,theta,U_indiv = model.fit_em(
                    iter=200, tol=0.1,
                    fit_emission=True,
                    fit_arrangement=False,
                    first_evidence=False)
                U_indiv = model.remap_evidence(U_indiv)
            else:
                # If no testset split, then use the U_indiv from training
                test_indx = np.ones((tinfo.shape[0],), dtype=bool)
                trainsess = [idx for idx in eval(minfo.sess) if isinstance(idx, list)][0]
                traind, _, _ = get_dataset(base_dir, test_data,
                                           atlas='MNISymC3', sess=trainsess)
                model.initialize([traind])
                U_indiv, _ = model.Estep()
                U_indiv = model.remap_evidence(U_indiv)

            # ------------------------------------------
            # Now run the DCBC evaluation fo the group
            Pgroup = pt.argmax(Prop, dim=0) + 1  # Get winner take all
            Pindiv = pt.argmax(U_indiv, dim=1) + 1  # Get winner take
            dcbc_indiv = calc_test_dcbc(Pindiv,tdata[:,test_indx,:], dist)
            dcbc_group = calc_test_dcbc(Pgroup,tdata[:,test_indx,:], dist)

            # ------------------------------------------
            # Collect the information from the evaluation
            # in a data frame
            ev_df = pd.DataFrame({'model_name':[minfo['name']]*num_subj,
                            'atlas':[minfo.atlas]*num_subj,
                            'K':[minfo.K]*num_subj,
                            'train_data':[minfo.datasets]*num_subj,
                            'train_loglik':[minfo.loglik]*num_subj,
                            'test_data':[test_data]*num_subj,
                            'indivtrain_ind':[indivtrain_ind]*num_subj,
                            'indivtrain_val':[indivtrain_values[n]]*num_subj,
                            'subj_num':np.arange(num_subj),
                            'common_kappa':[model.emissions[0].uniform_kappa]*num_subj})
            # Add all the evaluations to the data frame
            ev_df['dcbc_group']=dcbc_group.cpu()
            ev_df['dcbc_indiv']=dcbc_indiv.cpu()
            this_res = pd.concat([this_res, ev_df], ignore_index=True)

        # Concate model type
        this_res['model_type'] = i
        results = pd.concat([results, this_res], ignore_index=True)

    return results


def eval_all_prederror(model_type,prefix,K):
    models = ['Md','Po','Ni','Ib','MdPoNiIb']
    datasets = ['Ibc','Mdtb','Pontine','Nishimoto']

    model_name = []
    results = pd.DataFrame()
    for m in models:
        model_name.append(prefix + '_' +
                          m + '_' +
                          'space-MNISymC3'+ '_' +
                          f'K-{K}')
    for ds in datasets:
        print(f'Testdata: {ds}\n')
        R = run_prederror(model_type,model_name,ds,'all',
                    cond_ind=None,
                    part_ind='half',
                    eval_types=['group','floor'],
                    indivtrain_ind='half',indivtrain_values=[1,2])
        results = pd.concat([results,R],ignore_index=True)
    fname = base_dir + f'/Models/Evaluation_{model_type}/eval_prederr_{prefix}_K-{K}.tsv'
    results.to_csv(fname,sep='\t',index=False)

def eval_all_dcbc(model_type,prefix,K,space = 'MNISymC3', models=None, fname_suffix = None):
    """ Calculates DCBC over all models. 

        Args:
        model_type (str): Name of model type
        prefix (str): Name of test data set
        K (int): List or sessions to include into test_data
        space (str): Fieldname of the condition vector in test-data info
        models (str): List of models run on different training sets to evaluate.
            Defaults to None.
        fname_suffix (str): If given, results will be saved as tsv file with suffix appended.
             Specify if wanting to avoid overwriting old results. Defaults to None.

    """
    if models is None:
        models = ['Md','Po','Ni','Ib','Hc','MdPoNiIb','MdPoNiIbHc']
    datasets = ['MDTB','Pontine','Nishimoto']


    model_name = []
    results = pd.DataFrame()
    for m in models:
        model_name.append(prefix + '_' + m + '_' +
                          f'space-{space}'+ '_' + f'K-{K}')
    for ds in datasets:
        print(f'Testdata: {ds}\n')
        R = run_dcbc_individual(model_name,ds,'all',cond_ind=None,
                                part_ind='half',indivtrain_ind='half',
                                indivtrain_values=[1,2])
        results = pd.concat([results,R],ignore_index=True)

    prefix = '_'.join(models)
    fname = model_dir + f'/Models/Evaluation_{model_type}/eval_dcbc_{prefix}_K-{K}.tsv'
    if fname_suffix is not None:
        # Append fname suffix to avoid overwriting old results
        fname = fname.strip('.tsv') + f'_{fname_suffix}.tsv'
    results.to_csv(fname,sep='\t',index=False)
    print(f'Evaluation finished. Saved evaluation results in {fname}')



def eval_old_dcbc(models=None, datasets=None, fname_suffix=None):
    """ Evaluates old and new parcellations using new DCBC
    """
    parcels = ['Anatom','MDTB10','Buckner7','Buckner17','Ji10']
    if models is None:
        models = ['Models_01/asym_Md_space-MNISymC3_K-10.pickle']
    if datasets is None:
        datasets = ['Mdtb']

    par_name = []
    for p in parcels:
        par_name.append(base_dir + '/Atlases/tpl-MNI152NLin2009cSymC/' +
                f'atl-{p}_space-MNI152NLin2009cSymC_dseg.nii')
    par_name = models + par_name
    results = pd.DataFrame()
    for ds in datasets:
        print(f'Testdata: {ds}\n')
        R = run_dcbc_group(par_name,
                         space='MNISymC3',
                         test_data = ds,
                         test_sess = 'all')
        results = pd.concat([results,R],ignore_index=True)
    fname = base_dir + f'/Models/eval_dcbc_group.tsv'
    if fname_suffix is not None:
        # Append fname suffix to avoid overwriting old results
        fname = fname.strip('.tsv') + f'_{fname_suffix}.tsv'
    results.to_csv(fname,sep='\t',index=False)

def eval_ibc_prederror():
    # Make IBC session model file names
    # fnames = []
    # sess = DataSetIBC(base_dir + '/IBC').sessions
    # for s in sess:
    #     fnames.append(f'Models_05/asym_Ib_space-MNISymC3_K-10_{s}')
    #
    # fnames_group = ['Models_01/asym_Ib_space-MNISymC3_K-10',
    #                 'Models_02/asym_Ib_space-MNISymC3_K-10',
    #                 'Models_03/asym_Ib_space-MNISymC3_K-10',
    #                 'Models_04/asym_Ib_space-MNISymC3_K-10',
    #                 'Models_05/asym_Ib_space-MNISymC3_K-10']
    # model_name = fnames + fnames_group
    model_name = ['Models_02/asym_Ib_space-MNISymC3_K-10_ses-preference',
                  'Models_02/asym_Ib_space-MNISymC3_K-10_ses-rsvplanguage',
                  'Models_04/asym_Ib_space-MNISymC3_K-10_ses-preference+rsvplanguage']
    datasets = ['Mdtb','Pontine','Nishimoto', 'Ibc']
    if not isinstance(model_name,list):
        model_name = [model_name]

    results = pd.DataFrame()
    for ds in datasets:
        print(f'Testdata: {ds}\n')
        R = run_prederror(fnames_group,ds,'all',cond_ind=None,
                          part_ind='half', eval_types=['group','floor'],
                          indivtrain_ind='half',indivtrain_values=[1,2])
        results = pd.concat([results,R],ignore_index=True)
    fname = model_dir + f'/Models/Evaluation/eval_prederr_asym_Ib_K-10.tsv'
    results.to_csv(fname,sep='\t',index=False)

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

def run_ibc_sessfusion_indv_dcbc(sess1='preference', sess2='rsvplanguage'):
    ############# Evaluating IBC dataset sessions and plot results #############
    # remove the sessions were used to training to make test sessions
    sess = DataSetIBC(base_dir + '/IBC').sessions
    sess.remove('ses-' + sess1)
    sess.remove('ses-' + sess2)

    # Run DCBC individual evaluation of the sess1, sess2, sess12fusion
    # model_name = [f'Models_03/asym_Ib_space-MNISymC3_K-10_ses-{sess1}',
    #               f'Models_03/asym_Ib_space-MNISymC3_K-10_ses-{sess2}',
    #               f'Models_03/IBC_sessFusion/asym_Ib_space-MNISymC3_K-10_ses-{sess1}+{sess2}',
    model_name = [f'Models_02/asym_Ib_space-MNISymC3_K-10_ses-{sess1}',
                  f'Models_02/asym_Ib_space-MNISymC3_K-10_ses-{sess2}',
                  f'Models_04/IBC_sessFusion/asym_Ib_space-MNISymC3_K-10_ses-{sess1}+{sess2}']

    # Run DCBC individual
    res = run_dcbc_individual(model_name, 'IBC', sess, cond_ind=None,
                              part_ind='half', indivtrain_ind='half',
                              indivtrain_values=[1,2])

    oname = model_dir + f'/Models/Evaluation/eval_Ib_{sess1}+' \
                        f'{sess2}_dcbc_indv_on_leftsessions_SK.tsv'
    res.to_csv(oname, index=False, sep='\t')

    sb.barplot(data=res, x='model_type', y='dcbc_indiv',
               hue='common_kappa', errorbar="se")
    plt.show()

def eval_ibc_sessfusion_prederror(sess1='preference', sess2='rsvplanguage'):
    # remove the sessions were used to training to make test sessions
    sess = DataSetIBC(base_dir + '/IBC').sessions
    sess.remove('ses-' + sess1)
    sess.remove('ses-' + sess2)

    # model_name = [f'Models_03/asym_Ib_space-MNISymC3_K-10_ses-{sess1}',
    #               f'Models_03/asym_Ib_space-MNISymC3_K-10_ses-{sess2}',
    #               f'Models_03/IBC_sessFusion/asym_Ib_space-MNISymC3_K-10_ses-{sess1}+{sess2}']
    model_name = [f'Models_02/asym_Ib_space-MNISymC3_K-10_ses-{sess1}',
                  f'Models_02/asym_Ib_space-MNISymC3_K-10_ses-{sess2}',
                  f'Models_04/IBC_sessFusion/asym_Ib_space-MNISymC3_K-10_ses-{sess1}+{sess2}']
    datasets = ['Ibc']
    if not isinstance(model_name,list):
        model_name = [model_name]

    results = pd.DataFrame()
    for ds in datasets:
        print(f'Testdata: {ds}\n')
        R = run_prederror(model_name,ds,sess,cond_ind=None,
                          part_ind='half', eval_types=['group','floor'],
                          indivtrain_ind='half',indivtrain_values=[1,2])
        results = pd.concat([results,R],ignore_index=True)
    fname = model_dir + f'/Models/Evaluation/eval_Ib_{sess1}' \
                        f'+{sess2}_coserr_indv_on_leftsessions_SK.tsv'
    results.to_csv(fname,sep='\t',index=False)

    sb.barplot(data=results, x='model_type', y='coserr_ind3',
               hue='common_kappa', errorbar="se")
    plt.show()

def concat_eval(model_type,prefix,outfile):
    D = pd.DataFrame()
    for m in model_type:
        fname = model_dir + f'/Models/Evaluation/{m}.tsv'
        T = pd.read_csv(fname, delimiter='\t')
        D = pd.concat([D,T], ignore_index=True)
    oname = model_dir + f'/Models/Evaluation/eval_{prefix}_{outfile}.tsv'
    D.to_csv(oname,index=False,sep='\t')

    pass

def plot_all_sessFusion():
    from learn_fusion_cortex import fit_two_IBC_sessions
    sess_1 = DataSetIBC(base_dir + '/IBC').sessions
    sess_2 = DataSetIBC(base_dir + '/IBC').sessions

    for s1 in sess_1:
        sess_2.remove(s1)
        for s2 in sess_2:
            this_s1 = s1.split('-')[1]
            this_s2 = s2.split('-')[1]
            fname1 = model_dir + f'/Models/Evaluation/eval_Ib_{this_s1}+' \
                                f'{this_s2}_dcbc_indv_on_leftsessions_CK.tsv'
            fname2 = model_dir + f'/Models/Evaluation/eval_Ib_{this_s1}+' \
                                 f'{this_s2}_coserr_indv_on_leftsessions_CK.tsv'
            T1 = pd.read_csv(fname1, delimiter='\t')
            T2 = pd.read_csv(fname2, delimiter='\t')
            D1_dcbc = T1['dcbc_indiv'][T1['model_type'] == 0].mean()
            D2_dcbc = T1['dcbc_indiv'][T1['model_type'] == 1].mean()
            D_fusion_dcbc = T1['dcbc_indiv'][T1['model_type'] == 2].mean()

            D1_cos = T2['coserr_ind3'][T2['model_type'] == 0].mean()
            D2_cos = T2['coserr_ind3'][T2['model_type'] == 1].mean()
            D_fusion_cos = T2['coserr_ind3'][T2['model_type'] == 2].mean()

            if (D_fusion_dcbc < max(D1_dcbc, D2_dcbc)) and (D_fusion_cos > min(D1_cos, D2_cos)):
                # print(f'- Found common kappa {s1} and {s2}.')
                # Look at separate kappas
                fname1_sk = model_dir + f'/Models/Evaluation/eval_Ib_{this_s1}+' \
                                     f'{this_s2}_dcbc_indv_on_leftsessions_SK.tsv'
                fname2_sk = model_dir + f'/Models/Evaluation/eval_Ib_{this_s1}+' \
                                     f'{this_s2}_coserr_indv_on_leftsessions_SK.tsv'
                T1 = pd.read_csv(fname1_sk, delimiter='\t')
                T2 = pd.read_csv(fname2_sk, delimiter='\t')
                D1_dcbc_sk = T1['dcbc_indiv'][T1['model_type'] == 0].mean()
                D2_dcbc_sk = T1['dcbc_indiv'][T1['model_type'] == 1].mean()
                D_fusion_dcbc_sk = T1['dcbc_indiv'][T1['model_type'] == 2].mean()

                D1_cos_sk = T2['coserr_ind3'][T2['model_type'] == 0].mean()
                D2_cos_sk = T2['coserr_ind3'][T2['model_type'] == 1].mean()
                D_fusion_cos_sk = T2['coserr_ind3'][T2['model_type'] == 2].mean()

                if (D_fusion_dcbc_sk > max(D1_dcbc_sk, D2_dcbc_sk)) and (D_fusion_cos_sk < min(
                        D1_cos_sk, D2_cos_sk)):
                    print(f'- Found separate kappa {s1} and {s2}.')
                    # print(D1_dcbc, D1_dcbc_sk, D2_dcbc, D2_dcbc_sk, D_fusion_dcbc, D_fusion_dcbc_sk)
                    # print(D1_cos, D1_cos_sk, D2_cos, D2_cos_sk, D_fusion_cos, D_fusion_cos_sk)

def plot_figure():
    concat_eval(['eval_Ib_clips4+hcp2_dcbc_indv_on_leftsessions_CK',
                 'eval_Ib_clips4+hcp2_dcbc_indv_on_leftsessions_SK'], 'dcbc', 'clips4+hcp2')
    concat_eval(['eval_Ib_clips4+hcp2_coserr_indv_on_leftsessions_CK',
                 'eval_Ib_clips4+hcp2_coserr_indv_on_leftsessions_SK'], 'coserr', 'clips4+hcp2')
    filename = model_dir + f'/Models/Evaluation/eval_dcbc_clips4+hcp2.tsv'
    T = pd.read_csv(filename, delimiter='\t')
    sb.barplot(data=T, x='model_type', y='dcbc_indiv',
               hue='common_kappa', hue_order=T['common_kappa'].unique(), errorbar="se")
    plt.show()
    filename = model_dir + f'/Models/Evaluation/eval_coserr_clips4+hcp2.tsv'
    T = pd.read_csv(filename, delimiter='\t')
    sb.barplot(data=T, x='model_type', y='coserr_ind3',
               hue='common_kappa', hue_order=T['common_kappa'].unique(), errorbar="se")
    plt.ylim(0.8, 0.9)
    plt.show()

if __name__ == "__main__":
    # eval_old_dcbc(models=['Models_01/asym_MdPoNiIb_space-MNISymC3_K-10.pickle',
    # 'Models_01/asym_MdPoNiIbHc_space-MNISymC3_K-10.pickle'], datasets = ['Mdtb', 'Pontine', 'Nishimoto', 'Ibc'], fname_suffix='all')
    ## Evaluate IBC all sessions fusion on other datasets
    eval_all_dcbc('04', 'Models_04/asym', 20, space='MNISymC3', models=['Ib'], fname_suffix=None)

    ############# Evaluating IBC dataset sessions and plot results #############
    # Make IBC session model file names
    # fnames = []
    # sess = DataSetIBC(base_dir + '/IBC').sessions
    # sess.remove('ses-preference')
    # sess.remove('ses-rsvplanguage')
    # for s in sess:
    #     fnames.append(f'Models_05/asym_Ib_space-MNISymC3_K-10_{s}.pickle')
    #
    fnames_group = ['Models_01/asym_Md_space-MNISymC3_K-10_ses-s1',
                    'Models_02/asym_Md_space-MNISymC3_K-10_ses-s1_VMF_regkappa',
                    'Models_03/asym_Md_space-MNISymC3_K-10_ses-s1',
                    'Models_04/asym_Md_space-MNISymC3_K-10_ses-s1_VMF_regkappa',
                    'Models_05/asym_Md_space-MNISymC3_K-10_ses-s1_VMF_regkappa']

    # # 1. Run DCBC group evaluation
    # res = run_dcbc_group(fnames_group, 'MNISymC3', 'MDTB', test_sess=['ses-s2'],
    #                      saveFile='Evaluation/eval_Md_ses-s1_dcbc_group_on_ses-s2')
    res = run_dcbc_individual(fnames_group, 'MDTB', ['ses-s2'], cond_ind=None,
                              part_ind=None, indivtrain_ind=None,
                              indivtrain_values=[0])
    res = run_prederror(fnames_group, 'MDTB', ['ses-s2'], cond_ind=None,
                        part_ind=None, eval_types=['group', 'floor'],
                        indivtrain_ind=None, indivtrain_values=[1, 2])

    # sb.barplot(x='fit_type', y='dcbc', data=res)
    # plt.show()

    # plot_all_sessFusion()
    # 2. Run DCBC group and individual on IBC sess fusion
    sess_1 = DataSetIBC(base_dir + '/IBC').sessions
    sess_2 = DataSetIBC(base_dir + '/IBC').sessions

    for s1 in sess_1:
        sess_2.remove(s1)
        for s2 in sess_2:
            # run_ibc_sessfusion_indv_dcbc(sess1=s1.split('-')[1],
            #                              sess2=s2.split('-')[1])
            eval_ibc_sessfusion_prederror(sess1=s1.split('-')[1],
                                          sess2=s2.split('-')[1])
            print(f'-Done common kappa coserr evaluation fusion {s1} and {s2}.')

    # run_ibc_sessfusion_group_dcbc()

    # 3. Run adjusted expected cosine error
    eval_ibc_sessfusion_prederror(sess1='clips4', sess2='lyon1')

    pass
