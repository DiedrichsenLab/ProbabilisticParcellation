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
from ProbabilisticParcellation.util import *
import torch as pt
from learn_mdtb import get_mdtb_parcel
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import pickle
from copy import deepcopy

# Find model directory to save model fitting results
model_dir = 'Y:\data\Cerebellum\ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/robabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/robabilisticParcellationModel'
if not Path(model_dir).exists():
    raise(NameError('Could not find model_dir'))

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))
    
atlas_dir = base_dir + f'/Atlases'

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
    if isinstance(atlas, am.AtlasSurfaceSymmetric):
        P_arrange = atlas.Psym
        K_arrange = np.ceil(K / 2).astype(int)
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
            em_model = em.MixVMF(K=K, P=atlas.P,
                                 X=matrix.indicator(cond_vec[j]),
                                 part_vec=part_vec[j],
                                 uniform_kappa=uniform_kappa)
        else:
            raise((NameError(f'unknown emission model:{emission}')))
        em_models.append(em_model)

    # Make a full fusion model
    # TODO: cortical symmetric?
    if isinstance(atlas,am.AtlasSurfaceSymmetric):
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

    # Make the atlas object
    atlas_asym = am.get_atlas('fs32k', atlas_dir)
    bm_name = ['cortex_left', 'cortex_right']
    mask = []
    for i, hem in enumerate(['L', 'R']):
        mask.append(atlas_dir + f'/tpl-fs32k/tpl-fs32k_hemi-{hem}_mask.label.gii')
    atlas_sym = am.AtlasSurfaceSymmetric('fs32k', mask_gii=mask, structure=bm_name)

    atlas = [atlas_asym, atlas_sym]
    # Give a overall name for the type of model
    mname =['asym', 'sym']

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
    
    for i in [0, 1]:
        name = mname[i] + '_' + ''.join(dataname) 
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
        fname = f'/{name}_space-{atlas[i].name}_K-{K}'
        info.to_csv(wdir + fname + '.tsv',sep='\t')
        with open(wdir + fname + '.pickle','wb') as file:
            pickle.dump(models,file)

def clear_models(K,model_type='04'):
    for t in ['sym','asym']:
        for k in K:
            for s in ['MdPoNiIbHc_00','MdPoNiIbHc_02','MdPoNiIbHc_10']: # Md','Po','Ni','Hc','Ib','MdPoNiIb','MdPoNiIbHc','MdPoNiIbHc_00']:
                fname = f"Models_{model_type}/{t}_{s}_space-MNISymC3_K-{k}"
                try:
                    clear_batch(fname)
                    print(f"cleared {fname}")
                except:
                    print(f"skipping {fname}")

def write_dlabel_cifti(data, atlas,
                       labels=None,
                       label_names=None,
                       column_names=None,
                       label_RGBA=None):
    """Generates a label Cifti2Image from a numpy array

    Args:
        data (np.array):
            num_vert x num_col data
        atlas (obejct):
            the cortical surface <atlasSurface> object
        labels (list): Numerical values in data indicating the labels -
            defaults to np.unique(data)
        label_names (list):
            List of strings for names for labels
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors for labels
    Returns:
        gifti (GiftiImage): Label gifti image
    """
    if type(data) is pt.Tensor:
        data = data.numpy()

    if data.ndim == 1:
    # reshape to (1, num_vertices)
        data = data.reshape(-1,1)

    num_verts, num_cols = data.shape
    if labels is None:
        labels = np.unique(data)
    num_labels = len(labels)

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if label_RGBA is None:
        label_RGBA = [(0.0, 0.0, 0.0, 0.0)]
        if 0 in labels:
            num_labels -= 1
        hsv = plt.cm.get_cmap('hsv',num_labels)
        color = hsv(np.linspace(0,1,num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        for i in range(num_labels):
            label_RGBA.append((color[i][0],
                               color[i][1],
                               color[i][2],
                               color[i][3]))

    # Create label names from numerical values
    if label_names is None:
        label_names = ['???']
        for i in labels:
            if i == 0:
                pass
            else:
                label_names.append("label-{:02d}".format(i))

    assert len(label_RGBA) == len(label_names), \
        "The number of parcel labels must match the length of colors!"
    labelDict = []
    for i, nam in enumerate(label_names):
        labelDict.append((nam,label_RGBA[i]))

    labelAxis = nb.cifti2.LabelAxis(column_names, dict(enumerate(labelDict)))
    header = nb.Cifti2Header.from_axes((labelAxis, atlas.get_brain_model_axis()))
    img = nb.Cifti2Image(dataobj=data.reshape(1,-1), header=header)

    return img

if __name__ == "__main__":
    # fit_all([0], 10, model_type='04')

    info, model = load_batch_best('Models_04/asym_Md_space-fs32k_K-10')
    Prop = model.marginal_prob()
    par = pt.argmax(Prop, dim=0) + 1
    img = write_dlabel_cifti(par, am.get_atlas('fs32k', atlas_dir))
    nb.save(img, model_dir + f'/Models/Models_04/asym_Md_space-fs32k_K-10.dlabel.nii')
    # fit_all([1])
    # fit_all([2])
    # fit_all([0,1,2])
    # fit_all([0,1])
    # fit_all([0, 2]) 
    # fit_all([1, 2])
    # for k in [34]:
    #     fit_all([3],k,model_type='04')
        # fit_all([4],k,model_type='02')
        # fit_all([0,1,2,3,4],k,model_type='02')
    # fit_all([0],20)
    # fit_all([1],20)
    # fit_all([2],20)
    # fit_all([3],20)
    # check_IBC()
    #mask = base_dir + '/Atlases/tpl-MNI152NLIn2000cSymC/tpl-MNISymC_res-3_gmcmask.nii'
    #atlas = am.AtlasVolumetric('MNISymC3',mask_img=mask)

    #sess = [['ses-s1'],['ses-01'],['ses-01','ses-02']]
    #design_ind= ['cond_num_uni','task_id',',..']
    #info,models,Prop,V = load_batch_fit('asym_Md','MNISymC3',10)
    # parcel = pt.argmax(Prop,dim=1) # Get winner take all 
    # parcel=parcel[:,sym_atlas.indx_reduced] # Put back into full space
    # plot_parcel_flat(parcel[0:3,:],atlas,grid=[1,3],map_space='MNISymC') 
    # pass
    # pass
    # Prop, V = fit_niter(data,design,K,n_iter)
    # r1 = ev.calc_consistency(Prop,dim_rem=0)
    # r2 = ev.calc_consistency(V[0],dim_rem=2)


    # parcel = pt.argmax(Prop,dim=1)
    # plot_parcel_flat(parcel,suit_atlas,(1,4))
    
                

    pass
