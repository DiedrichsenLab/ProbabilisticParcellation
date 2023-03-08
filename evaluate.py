# Evaluate cerebellar parcellations
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
import Functional_Fusion.dataset as ds
import generativeMRF.emissions as em
import generativeMRF.arrangements as ar
import generativeMRF.full_model as fm
import generativeMRF.evaluation as ev
import ProbabilisticParcellation.util as ut

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
######################################################
# The new GPU capatible DCBC evaluation function is now
# callable in util.py. If you prefer use CPU version, please
# uncomment below import line (highly not-recommend)
######################################################
# from DCBC.DCBC_vol import compute_DCBC, compute_dist


# Find model directory to save model fitting results
model_dir = 'Y:\data\Cerebellum\ProbabilisticParcellationModel'
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
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise (NameError('Could not find base_dir'))

atlas_dir = base_dir + f'/Atlases'


def calc_test_error(M, tdata, U_hats):
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
    pred_err = np.empty((len(U_hats), num_subj))
    for s in range(num_subj):
        print(f'Subject:{s}', end=':')
        tic = time.perf_counter()
        # initialize the emssion model using all but one subject
        M.emissions[0].initialize(tdata[subj != s, :, :])
        # For fitting an emission model witout the arrangement model,
        # We can not without multiple starting values
        M.initialize()
        M, ll, theta, Uhat = M.fit_em(
            iter=200, tol=0.1,
            fit_emission=True,
            fit_arrangement=False,
            first_evidence=False)
        X = M.emissions[0].X
        dat = pt.linalg.pinv(X) @ tdata[subj == s, :, :]
        for i, crit in enumerate(U_hats):
            if crit == 'group':
                U = group_parc
            elif crit == 'floor':
                # U,ll = M.Estep(Y=pt.tensor(tdata[subj==s,:,:]).unsqueeze(0))
                M.emissions[0].initialize(tdata[subj == s, :, :])
                U = pt.softmax(M.emissions[0].Estep(
                    tdata[subj == s, :, :]), dim=1)
            elif crit.ndim == 2:
                U = crit
            elif crit.ndim == 3:
                U = crit[subj == s, :, :]
            else:
                raise (
                    NameError("U_hats needs to be 'group','floor',a 2-d or 3d-tensor"))
            a = ev.coserr(dat, M.emissions[0].V, U,
                          adjusted=True, soft_assign=True)
            pred_err[i, s] = a
        toc = time.perf_counter()
        print(f"{toc - tic:0.4f}s")
    return pred_err


def calc_test_dcbc(parcels, testdata, dist, max_dist=110, bin_width=5,
                   trim_nan=False, verbose=True):
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
        if verbose:
            print(f'Subject {sub}', end=':')
        tic = time.perf_counter()
        if parcels.ndim == 1:
            D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                             parcellation=parcels,
                             dist=dist, func=testdata[sub].T)
        else:
            D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                             parcellation=parcels[sub],
                             dist=dist, func=testdata[sub].T)
        dcbc_values.append(D['DCBC'])
        toc = time.perf_counter()
        if verbose:
            print(f"{toc-tic:0.4f}s")
    return pt.stack(dcbc_values)


def run_prederror(model_names, test_data, test_sess, cond_ind,
                  part_ind=None, eval_types=['group', 'floor'],
                  indivtrain_ind=None, indivtrain_values=[0],
                  device=None, load_best=True):
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
    tdata, tinfo, tds = ds.get_dataset(base_dir, test_data,
                                       atlas='MNISymC3', sess=test_sess)
    # convert tdata to tensor
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

    # For testing: tdata=tdata[0:5,:,:]
    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    if not isinstance(model_names, list):
        model_names = [model_names]

    # Get condition and partition vector of test data
    if cond_ind is None:
        cond_ind = tds.cond_ind
    cond_vec = tinfo[cond_ind].values.reshape(-1,)
    if part_ind is None:
        part_vec = np.zeros((tinfo.shape[0],), dtype=int)
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
        if load_best:
            minfo, model = load_batch_best(f"{model_name}", device=device)
        else:
            minfo, model = load_batch_fit(f"{model_name}")
            minfo = minfo.iloc[0]

        model_kp = model.emissions[0].uniform_kappa
        this_res = pd.DataFrame()
        # Loop over the splits - if split then train a individual model
        for n in range(n_splits):
            # ------------------------------------------
            # Train an emission model on the individual training data
            # and get a Uhat (individual parcellation) from it.
            if indivtrain_ind is not None:
                train_indx = tinfo[indivtrain_ind] == indivtrain_values[n]
                test_indx = tinfo[indivtrain_ind] != indivtrain_values[n]
                indivtrain_em = em.MixVMF(K=minfo.K, N=40,
                                          P=model.emissions[0].P,
                                          X=matrix.indicator(
                                              cond_vec[train_indx]),
                                          part_vec=part_vec[train_indx],
                                          uniform_kappa=model_kp)
                indivtrain_em.initialize(tdata[:, train_indx, :])
                model.emissions = [indivtrain_em]
                model.initialize()
                m, ll, theta, U_indiv = model.fit_em(iter=200, tol=0.1,
                                                     fit_emission=True,
                                                     fit_arrangement=False,
                                                     first_evidence=False)
                # Add individual U_hat data (emission) only
                Uhat_em = pt.softmax(m.emissions[0].Estep(), dim=1)
                # Uhat_compl, _ = m.arrange.Estep(m.emissions[0].Estep())
                all_eval = eval_types + [Uhat_em] + \
                    [model.remap_evidence(U_indiv)]
            else:
                test_indx = np.ones((tinfo.shape[0],), dtype=bool)
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
            res = calc_test_error(model, tdata[:, test_indx, :], all_eval)
            # ------------------------------------------
            # Collect the information from the evaluation
            # in a data frame
            train_datasets = minfo.datasets
            if isinstance(minfo.datasets, pd.Series):
                train_datasets = minfo.datasets.tolist()
            ev_df = pd.DataFrame({'model_name': [minfo['name']] * num_subj,
                                  'atlas': [minfo.atlas] * num_subj,
                                  'K': [minfo.K] * num_subj,
                                  'train_data': [train_datasets] * num_subj,
                                  'train_loglik': [minfo.loglik] * num_subj,
                                  'test_data': [test_data] * num_subj,
                                  'indivtrain_ind': [indivtrain_ind] * num_subj,
                                  'indivtrain_val': [indivtrain_values[n]] * num_subj,
                                  'subj_num': np.arange(num_subj),
                                  'common_kappa': [model_kp] * num_subj})
            # Add all the evaluations to the data frame
            for e, ev in enumerate(all_eval):
                if isinstance(ev, str):
                    ev_df['coserr_' + ev] = res[e, :]
                else:
                    ev_df[f'coserr_ind{e}'] = res[e, :]
            this_res = pd.concat([this_res, ev_df], ignore_index=True)

        # Concate model type
        this_res['model_type'] = model_name.split('/')[0]
        # Add a column it's session fit
        if len(model_name.split('ses-')) >= 2:
            this_res['test_sess'] = model_name.split('ses-')[1]
        else:
            this_res['test_sess'] = 'all'
        results = pd.concat([results, this_res], ignore_index=True)

    return results


def run_dcbc_group(par_names, space, test_data, test_sess='all', saveFile=None,
                   device=None, tdata=None, verbose=True):
    """ Run DCBC group evaluation

    Args:
        par_names (list): List of names for the parcellations to evaluate    
                Can be either 
                    nifti files (*_dseg.nii) or 
                    models (*.npy)
        space (str): Atlas space (SUIT3, MNISym3C)... 
        test_data (str): Data set string 
        test_sess (str, optional): Data set test. Defaults to 'all'.
        tdata (np.array, optional): Data to use for testing. Defaults to None. Use when running dcbc subject by subject for timeseries evaluation if loading all subjects is not feasible.

    Returns:
        DataFrame: Results
    """
    if tdata is None:
        tdata, _, _ = ds.get_dataset(base_dir, test_data,
                                     atlas=space, sess=test_sess, verbose=True)
    atlas, _ = am.get_atlas(space, atlas_dir=base_dir + '/Atlases')
    dist = compute_dist(atlas.world.T, resolution=1)

    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    if not isinstance(par_names, list):
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
        if pname_parts[-1] == 'pickle':
            minfo, model = load_batch_best(f"{fileparts[-2]}/{pname_parts[-2]}",
                                           device=device)
            Prop = model.marginal_prob()
            par = pt.argmax(Prop, dim=0) + 1
        elif pname_parts[-1] == 'nii':
            par = atlas.read_data(pn, 0)
            if not pt.is_tensor(par):
                par = pt.tensor(par)

        # Initialize result array
        if i == 0:
            dcbc = pt.zeros((len(par_names), tdata.shape[0]))
        print(f"Number zeros {(par==0).sum()}")
        dcbc[i, :] = calc_test_dcbc(par, tdata, dist, verbose=verbose)
        num_subj = tdata.shape[0]

        ev_df = pd.DataFrame({'fit_type': [fileparts[0]] * num_subj,
                              'model_name': [pname_parts[-2]] * num_subj,
                              'test_data': [test_data] * num_subj,
                              'subj_num': np.arange(num_subj),
                              'dcbc': dcbc[i, :].cpu().numpy()
                              })
        results = pd.concat([results, ev_df], ignore_index=True)

    if saveFile is not None:
        oname = model_dir + f'/Models/{saveFile}.tsv'
        results.to_csv(oname, index=False, sep='\t')

    return results


def run_dcbc(model_names, tdata, atlas, train_indx, test_indx, cond_vec,
             part_vec, device=None, load_best=True, verbose=True):
    """ Calculates DCBC using a test_data set. The test data splitted into
        individual training and test set given by `train_indx` and `test_indx`.
        First we use individual training data to derive an individual
        parcellations (using the model) and evaluate it on test data.
        By calling function `calc_test_dcbc`, the Means of the parcels are
        always estimated on N-1 subjects and evaluated on the Nth left-out
        subject.
    Args:
        model_names (list or str): Name of model fit (tsv/pickle file)
        tdata (pt.Tensor or np.ndarray): test data set
        atlas (atlas_map): The atlas map object for calculating voxel distance
        train_indx (ndarray of index or boolean mask): index of individual
            training data
        test_indx (ndarray or index boolean mask): index of individual test
            data
        cond_vec (1d array): the condition vector in test-data info
        part_vec (1d array): partition vector in test-data info
        device (str): the device name to load trained model
        load_best (str): I don't know
    Returns:
        data-frame with model evalution of both group and individual DCBC
    Notes:
        This function is modified for DCBC group and individual evaluation
        in general case (not include IBC two sessions evaluation senario)
        requested by Jorn.
    """
    # Calculate distance metric given by input atlas
    dist = compute_dist(atlas.world.T, resolution=1)
    # convert tdata to tensor
    if type(tdata) is np.ndarray:
        tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())

    if not isinstance(model_names, list):
        model_names = [model_names]

    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    # Now loop over possible models we want to evaluate
    for i, model_name in enumerate(model_names):
        print(f"Doing model {model_name}\n")
        if verbose:
            ut.report_cuda_memory()
        if load_best:
            minfo, model = load_batch_best(f"{model_name}", device=device)
        else:
            minfo, model = load_batch_fit(f"{model_name}")
            minfo = minfo.iloc[0]

        Prop = model.marginal_prob()
        this_res = pd.DataFrame()
        # ------------------------------------------
        # Train an emission model on the individual training data
        # and get a Uhat (individual parcellation) from it.
        indivtrain_em = em.MixVMF(K=minfo.K, N=40,
                                  P=model.emissions[0].P,
                                  X=matrix.indicator(cond_vec[train_indx]),
                                  part_vec=part_vec[train_indx],
                                  uniform_kappa=model.emissions[0].uniform_kappa)
        indivtrain_em.initialize(tdata[:, train_indx, :])
        model.emissions = [indivtrain_em]
        model.initialize()
        # Gets us the individual parcellation
        model, _, _, U_indiv = model.fit_em(iter=200, tol=0.1,
                                            fit_emission=True,
                                            fit_arrangement=False,
                                            first_evidence=False)
        U_indiv = model.remap_evidence(U_indiv)

        # ------------------------------------------
        # Now run the DCBC evaluation fo the group and individuals
        Pgroup = pt.argmax(Prop, dim=0) + 1
        Pindiv = pt.argmax(U_indiv, dim=1) + 1
        dcbc_group = calc_test_dcbc(
            Pgroup, tdata[:, test_indx, :], dist, verbose=verbose)
        dcbc_indiv = calc_test_dcbc(
            Pindiv, tdata[:, test_indx, :], dist, verbose=verbose)

        # ------------------------------------------
        # Collect the information from the evaluation
        # in a data frame
        train_datasets = minfo.datasets
        if isinstance(minfo.datasets, pd.Series):
            train_datasets = minfo.datasets.tolist()
        ev_df = pd.DataFrame({'model_name': [minfo['name']] * num_subj,
                              'atlas': [minfo.atlas] * num_subj,
                              'K': [minfo.K] * num_subj,
                              'train_data': [train_datasets] * num_subj,
                              'train_loglik': [minfo.loglik] * num_subj,
                              'subj_num': np.arange(num_subj),
                              'common_kappa': [model.emissions[0].uniform_kappa] * num_subj})
        # Add all the evaluations to the data frame
        ev_df['dcbc_group'] = dcbc_group.cpu()
        ev_df['dcbc_indiv'] = dcbc_indiv.cpu()
        this_res = pd.concat([this_res, ev_df], ignore_index=True)

        # Concate model type
        this_res['model_type'] = model_name.split('/')[0]
        # Add a column it's session fit
        if len(model_name.split('ses-')) >= 2:
            this_res['test_sess'] = model_name.split('ses-')[1]
        else:
            this_res['test_sess'] = 'all'
        results = pd.concat([results, this_res], ignore_index=True)

    return results


def run_dcbc_IBC(model_names, test_data, test_sess,
                 cond_ind=None, part_ind=None,
                 indivtrain_ind=None, indivtrain_values=[0],
                 device=None, load_best=True):
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
    tdata, tinfo, tds = ds.get_dataset(base_dir, test_data,
                                       atlas='MNISymC3', sess=test_sess)
    atlas, _ = am.get_atlas('MNISymC3', atlas_dir=base_dir + '/Atlases')
    dist = compute_dist(atlas.world.T, resolution=1)

    # convert tdata to tensor
    tdata = pt.tensor(tdata, dtype=pt.get_default_dtype())
    # For testing: tdata=tdata[0:5,:,:]
    num_subj = tdata.shape[0]
    results = pd.DataFrame()
    if not isinstance(model_names, list):
        model_names = [model_names]

    # Get condition vector of test data
    if cond_ind is None:
        # get default cond_ind from testdataset
        cond_vec = tinfo[tds.cond_ind].values.reshape(-1, )
    else:
        cond_vec = tinfo[cond_ind].values.reshape(-1, )

    # Get partition vector of test data
    if part_ind is None:
        part_vec = np.ones((tinfo.shape[0],), dtype=int)
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
        if load_best:
            minfo, model = load_batch_best(f"{model_name}", device=device)
        else:
            minfo, model = load_batch_fit(f"{model_name}")
            minfo = minfo.iloc[0]

        Prop = model.marginal_prob()

        this_res = pd.DataFrame()
        # Loop over the splits - if split then train a individual model
        for n in range(n_splits):
            # ------------------------------------------
            # Train an emission model on the individual training data
            # and get a Uhat (individual parcellation) from it.
            if indivtrain_ind is not None:
                train_indx = tinfo[indivtrain_ind] == indivtrain_values[n]
                test_indx = tinfo[indivtrain_ind] != indivtrain_values[n]
                indivtrain_em = em.MixVMF(K=minfo.K, N=40,
                                          P=model.emissions[0].P,
                                          X=matrix.indicator(
                                              cond_vec[train_indx]),
                                          part_vec=part_vec[train_indx],
                                          uniform_kappa=model.emissions[0].uniform_kappa)
                indivtrain_em.initialize(tdata[:, train_indx, :])
                model.emissions = [indivtrain_em]
                model.initialize()
                # Gets us the individual parcellation
                model, ll, theta, U_indiv = model.fit_em(
                    iter=200, tol=0.1,
                    fit_emission=True,
                    fit_arrangement=False,
                    first_evidence=False)
                U_indiv = model.remap_evidence(U_indiv)
            else:
                # If no testset split, then use the U_indiv from training
                test_indx = np.ones((tinfo.shape[0],), dtype=bool)
                trainsess = [idx for idx in eval(
                    minfo.sess) if isinstance(idx, list)][0]
                traind, info, _ = ds.get_dataset(base_dir, test_data,
                                                 atlas='MNISymC3', sess=trainsess)
                # Check if the model was trained joint or separate sessions
                if len(model.emissions) == 1:
                    model.initialize(
                        [np.hstack([traind[:, info.sess == s, :] for s in trainsess])])
                else:
                    model.initialize([traind[:, info.sess == s, :]
                                     for s in trainsess])

                # Get the individual parcellation on all training data
                model, _, _, U_indiv = model.fit_em(iter=200, tol=0.1,
                                                    fit_emission=True, fit_arrangement=False,
                                                    first_evidence=False)
                U_indiv = model.remap_evidence(U_indiv)

            # ------------------------------------------
            # Now run the DCBC evaluation fo the group
            Pgroup = pt.argmax(Prop, dim=0) + 1  # Get winner take all
            Pindiv = pt.argmax(U_indiv, dim=1) + 1  # Get winner take
            dcbc_indiv = calc_test_dcbc(Pindiv, tdata[:, test_indx, :], dist)
            dcbc_group = calc_test_dcbc(Pgroup, tdata[:, test_indx, :], dist)

            # ------------------------------------------
            # Collect the information from the evaluation
            # in a data frame
            ev_df = pd.DataFrame({'model_name': [minfo['name']] * num_subj,
                                  'atlas': [minfo.atlas] * num_subj,
                                  'K': [minfo.K] * num_subj,
                                  'emission': [minfo.emission] * num_subj,
                                  'train_data': [minfo.datasets] * num_subj,
                                  'train_loglik': [minfo.loglik] * num_subj,
                                  'test_data': [test_data] * num_subj,
                                  'indivtrain_ind': [indivtrain_ind] * num_subj,
                                  'indivtrain_val': [indivtrain_values[n]] * num_subj,
                                  'subj_num': np.arange(num_subj),
                                  'common_kappa': [model.emissions[0].uniform_kappa] * num_subj})
            # Add all the evaluations to the data frame
            ev_df['dcbc_group'] = dcbc_group.cpu()
            ev_df['dcbc_indiv'] = dcbc_indiv.cpu()
            this_res = pd.concat([this_res, ev_df], ignore_index=True)

        # Concate model type
        this_res['model_type'] = model_name.split('/')[0]
        # Add a column it's session fit
        if len(model_name.split('ses-')) >= 2:
            this_res['test_sess'] = model_name.split('ses-')[1]
        else:
            this_res['test_sess'] = 'all'
        results = pd.concat([results, this_res], ignore_index=True)

    return results


def eval_all_prederror(model_type, prefix, K, verbose=True):
    models = ['Md', 'Po', 'Ni', 'Ib', 'MdPoNiIb']
    datasets = ['Ibc', 'Mdtb', 'Pontine', 'Nishimoto']

    model_name = []
    results = pd.DataFrame()
    for m in models:
        model_name.append(prefix + '_' +
                          m + '_' +
                          'space-MNISymC3' + '_' +
                          f'K-{K}')
    for ds in datasets:
        if verbose:
            ut.report_cuda_memory()
        print(f'Testdata: {ds}\n')
        R = run_prederror(model_type, model_name, ds, 'all',
                          cond_ind=None,
                          part_ind='half',
                          eval_types=['group', 'floor'],
                          indivtrain_ind='half', indivtrain_values=[1, 2])
        results = pd.concat([results, R], ignore_index=True)
    fname = base_dir + \
        f'/Models/Evaluation_{model_type}/eval_prederr_{prefix}_K-{K}.tsv'
    results.to_csv(fname, sep='\t', index=False)


def eval_all_dcbc(model_type, prefix, K, space='MNISymC3', models=None, fname_suffix=None, verbose=True):
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
        models = ['Md', 'Po', 'Ni', 'Ib', 'Hc', 'MdPoNiIb', 'MdPoNiIbHc']
    datasets = ['MDTB', 'Pontine', 'Nishimoto']

    model_name = []
    results = pd.DataFrame()
    for m in models:
        model_name.append(prefix + '_' + m + '_' +
                          f'space-{space}' + '_' + f'K-{K}')
    for ds in datasets:
        print(f'Testdata: {ds}\n')
        if verbose:
            ut.report_cuda_memory()
        R = run_dcbc_individual(model_name, ds, 'all', cond_ind=None,
                                part_ind='half', indivtrain_ind='half',
                                indivtrain_values=[1, 2])
        results = pd.concat([results, R], ignore_index=True)

    prefix = '_'.join(models)
    fname = model_dir + \
        f'/Models/Evaluation_{model_type}/eval_dcbc_{prefix}_K-{K}.tsv'
    if fname_suffix is not None:
        # Append fname suffix to avoid overwriting old results
        fname = fname.strip('.tsv') + f'_{fname_suffix}.tsv'
    results.to_csv(fname, sep='\t', index=False)
    print(f'Evaluation finished. Saved evaluation results in {fname}')


def eval_old_dcbc(models=None, datasets=None, fname_suffix=None):
    """ Evaluates old and new parcellations using new DCBC
    """
    parcels = ['Anatom', 'MDTB10', 'Buckner7', 'Buckner17', 'Ji10']
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
                           test_data=ds,
                           test_sess='all')
        results = pd.concat([results, R], ignore_index=True)
    fname = base_dir + f'/Models/eval_dcbc_group.tsv'
    if fname_suffix is not None:
        # Append fname suffix to avoid overwriting old results
        fname = fname.strip('.tsv') + f'_{fname_suffix}.tsv'
    results.to_csv(fname, sep='\t', index=False)


def concat_all_prederror(model_type, prefix, K, outfile):
    D = pd.DataFrame()
    for p in prefix:
        for k in K:
            fname = base_dir + \
                f'/Models/Evaluation_{model_type}/eval_prederr_{p}_K-{k}.tsv'
            T = pd.read_csv(fname, delimiter='\t')
            T['prefix'] = [p] * T.shape[0]
            D = pd.concat([D, T], ignore_index=True)
    oname = base_dir + \
        f'/Models/Evaluation_{model_type}/eval_prederr_{outfile}.tsv'
    D.to_csv(oname, index=False, sep='\t')

    pass


if __name__ == "__main__":

    # model_type='04'
    # sym='asym'
    # fname_suffix='HCPw_asym'

    # # Evaluate DCBC
    # eval_all_dcbc(model_type=model_type,prefix=sym,K=K,space = 'MNISymC3', models=hcp_models, fname_suffix=fname_suffix)

    # Concat DCBC
    # concat_all_prederror(model_type=model_type,prefix=sym,K=Ks,outfile=fname_suffix)

    pass
