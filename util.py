import numpy as np
import nibabel as nb
import SUITPy as suit
import pickle
from pathlib import Path
import Functional_Fusion.atlas_map as am
import pandas as pd
import torch as pt
import json
import matplotlib.pyplot as plt
import generativeMRF.evaluation as ev
import generativeMRF.full_model as fm

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


def cal_corr(Y_target, Y_source):
    """ Matches the rows of two Y_source matrix to Y_target
    Using row-wise correlation and matching the highest pairs
    consecutively
    Args:
        Y_target: Matrix to align to
        Y_source: Matrix that is being aligned
    Returns:
        indx: New indices, so that YSource[indx,:]~=Y_target
    """
    K = Y_target.shape[0]
    # Compute the row x row correlation matrix
    Y_tar = Y_target - Y_target.mean(dim=1,keepdim=True)
    Y_sou = Y_source - Y_source.mean(dim=1,keepdim=True)
    Cov = pt.matmul(Y_tar, Y_sou.t())
    Var1 = pt.sum(Y_tar*Y_tar, dim=1)
    Var2 = pt.sum(Y_sou*Y_sou, dim=1)
    Corr = Cov / pt.sqrt(pt.outer(Var1, Var2))

    return Corr

def load_batch_fit(fname):
    """ Loads a batch of fits and extracts marginal probability maps 
    and mean vectors
    Args:
        fname (str): File name
    Returns: 
        info: Data Frame with information 
        models: List of models
    """
    wdir = model_dir + '/Models/'
    info = pd.read_csv(wdir + fname + '.tsv',sep='\t')
    with open(wdir + fname + '.pickle','rb') as file:
        models = pickle.load(file)
    return info,models

def clear_batch(fname):
    """Ensures that pickle file does not contain superflous data
    Args:
        fname (): filename
    """
    wdir = base_dir + '/Models/'
    with open(wdir + fname + '.pickle','rb') as file:
        models = pickle.load(file)
    # Clear models 
    for m in models:
        m.clear()
    
    with open(wdir + fname + '.pickle','wb') as file:
        pickle.dump(models,file)

def move_batch_to_device(fname, device='cpu'):
    """Overwrite all tensors in the batch fitted models
       from torch.cuda to the normal torch.Tensor for
       people who cannot use cuda.
    Args:
        fname (): filename
        device: the target device to store tensors
    """
    wdir = model_dir + '/Models/'
    with open(wdir + fname + '.pickle', 'rb') as file:
        models = pickle.load(file)

    # Recursively tensors to device
    for m in models:
        m.move_to(device=device)

    with open(wdir + fname + '.pickle', 'wb') as file:
        pickle.dump(models, file)

def load_batch_best(fname):
    """ Loads a batch of model fits and selects the best one
    Args:
        fname (str): File name
    """
    info, models = load_batch_fit(fname)
    j = info.loglik.argmax()

    best_model = models[j]

    return info.iloc[j], best_model

def get_colormap_from_lut(fname=base_dir + '/Atlases/tpl-SUIT/atl-MDTB10.lut'):
    """ Makes a color map from a *.lut file 
    Args:
        fname (str): Name of Lut file

    Returns:
        _type_: _description_
    """
    color_info = pd.read_csv(fname, sep=' ', header=None)
    color_map = np.zeros((color_info.shape[0]+1, 3))
    color_map = color_info.iloc[:, 1:4].to_numpy()
    return color_map


def plot_data_flat(data,atlas,
                    cmap = None,
                    dtype = 'label',
                    cscale = None,
                    labels = None,
                    render='matplotlib',
                    colorbar = False):
    """ Maps data from an atlas space to a full volume and
    from there onto the surface - then plots it. 

    Args:
        data (_type_): _description_
        atlas (str): Atlas code ('SUIT3','MNISymC3',...)
        cmap (_type_, optional): Colormap. Defaults to None.
        dtype (str, optional): 'label' or 'func'
        cscale (_type_, optional): Color scale 
        render (str, optional): 'matplotlib','plotly'

    Returns:
        ax: Axis / figure of plot
    """
    # Plot Data from a specific atlas space on the flatmap
    suit_atlas, _ = am.get_atlas(atlas,base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(data)
    
    # Figure out correct mapping space 
    if atlas[0:4]=='SUIT':
        map_space='SUIT'
    elif atlas[0:7]=='MNISymC':
        map_space='MNISymC'
    else:
        raise(NameError('Unknown atlas space'))

    # Plotting label 
    if dtype =='label':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='mode',
            space=map_space,ignore_zeros=True)
        ax = suit.flatmap.plot(surf_data, 
                render=render,
                cmap=cmap, 
                new_figure=False,
                label_names = labels,
                overlay_type='label',
                colorbar= colorbar)
    # Plotting funtional data 
    elif dtype== 'func':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='nanmean',
            space=map_space)
        ax = suit.flatmap.plot(surf_data, 
                render=render,
                cmap=cmap,
                cscale = cscale,
                new_figure=False,
                overlay_type='func',
                colorbar= colorbar)
    else:
        raise(NameError('Unknown data type'))
    return ax

def plot_multi_flat(data,atlas,grid,
                    cmap = None,
                    dtype = 'label',
                    cscale = None,
                    titles=None,
                    colorbar = False):
    """Plots a grid of flatmaps with some data 

    Args:
        data (array): NxP array of data 
        atlas (str): Atlas code ('SUIT3','MNISymC3',...)
        grid (tuple): (rows,cols) grid for subplot 
        cmap (colormap): Color map Defaults to None.
        dtype (str, optional):'label' or 'func'
        cscale (_type_, optional): Scale of data (None)
        titles (_type_, optional): _description_. Defaults to None.
    """
    for i in range(data.shape[0]):
        plt.subplot(grid[0],grid[1],i+1)
        plot_data_flat(data[i,:],atlas,
                    cmap = cmap,
                    dtype = dtype,
                    cscale = cscale,
                    render='matplotlib',
                    colorbar = (i==0) & colorbar)
        if titles is not None:
            plt.title(titles[i])
            plt.savefig(f'rel_{titles[i]}.png', format='png')

def plot_model_parcel(model_names,grid,cmap='tab20b',align=False):
    """  Load a bunch of model fits, selects the best from 
    each of them and plots the flatmap of the parcellation

    Args:
        model_names (list): List of mode names 
        grid (tuple): (rows,cols) of matrix 
        cmap (str / colormat): Colormap. Defaults to 'tab20b'.
        align (bool): Align the models before plotting. Defaults to False.
    """
    titles = [] 
    models = []

    # Load models and produce titles 
    for i,mn in enumerate(model_names):
        info,model = load_batch_best(mn)
        models.append(model)
        # Split the name and build titles
        fname = mn.split('/') # Get filename if directory is given 
        split_mn = fname[-1].split('_') 
        atlas = split_mn[2][6:]
        titles.append(split_mn[1] + ' ' + split_mn[3])
    
    # Align models if requested 
    if align:
        Prob = ev.align_models(models,in_place=False)
    else: 
        Prob = ev.extract_marginal_prob(models)

    if type(Prob) is pt.Tensor:
        if pt.cuda.is_available() or pt.backends.mps.is_built():
            Prob = Prob.cpu().numpy()
        else:
            Prob = Prob.numpy()

    parc = np.argmax(Prob,axis=1)+1


    plot_multi_flat(parc,atlas,grid=grid,
                     cmap=cmap,
                     titles=titles)

def _compute_var_cov(data, cond='all', mean_centering=True):
    """
        Compute the affinity matrix by given kernel type,
        default to calculate Pearson's correlation between all vertex pairs

        :param data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        :param cond: specify the subset of activation conditions to evaluation
                    (e.g condition column [1,2,3,4]),
                     if not given, default to use all conditions
        :param mean_centering: boolean value to determine whether the given subject data
                               should be mean centered

        :return: cov - the covariance matrix of current subject data. shape [N * N]
                 var - the variance matrix of current subject data. shape [N * N]
    """
    if mean_centering:
        data = data - pt.mean(data, dim=1, keepdim=True) # mean centering
    else:
        data = data

    # specify the condition index used to compute correlation, otherwise use all conditions
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")

    k = data.shape[1]
    cov = pt.matmul(data, data.T) / (k-1)
    sd = data.std(dim=1).reshape(-1,1)  # standard deviation
    var = pt.matmul(sd, sd.T)

    return cov, var

def compute_dist(coord, resolution=2):
    """
    calculate the distance matrix between each of the voxel pairs by given mask file

    :param coord: the ndarray of all N voxels coordinates x,y,z. Shape N * 3
    :param resolution: the resolution of .nii file. Default 2*2*2 mm

    :return: a distance matrix of N * N, where N represents the number of masked voxels
    """
    if type(coord) is np.ndarray:
        coord = pt.tensor(coord, dtype=pt.get_default_dtype())

    num_points = coord.shape[0]
    D = pt.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return pt.sqrt(D) * resolution

def compute_DCBC(maxDist=35, binWidth=1, parcellation=np.empty([]),
                 func=None, dist=None, weighting=True):
    """
    The main entry of DCBC calculation for volume space
    :param hems:        Hemisphere to test. 'L' - left hemisphere; 'R' - right hemisphere; 'all' - both hemispheres
    :param maxDist:     The maximum distance for vertices pairs
    :param binWidth:    The spatial binning width in mm, default 1 mm
    :param parcellation:
    :param dist_file:   The path of distance metric of vertices pairs, for example Dijkstra's distance, GOD distance
                        Euclidean distance. Dijkstra's distance as default
    :param weighting:   Boolean value. True - add weighting scheme to DCBC (default)
                                       False - no weighting scheme to DCBC
    """

    numBins = int(np.floor(maxDist / binWidth))

    cov, var = _compute_var_cov(func)
    # cor = np.corrcoef(func)

    # remove the nan value and medial wall from dist file
    dist = dist.to_sparse()
    row = dist.indices()[0]
    col = dist.indices()[1]
    distance = dist.values()
    # row, col, distance = sp.sparse.find(dist)

    # making parcellation matrix without medial wall and nan value
    par = parcellation
    num_within, num_between, corr_within, corr_between = [], [], [], []
    for i in range(numBins):
        inBin = pt.where((distance > i * binWidth) & (distance <= (i + 1) * binWidth))[0]

        # lookup the row/col index of within and between vertices
        within = pt.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
        between = pt.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

        # retrieve and append the number of vertices for within/between in current bin
        num_within.append(pt.tensor(within.numel(), dtype=pt.get_default_dtype()))
        num_between.append(pt.tensor(between.numel(), dtype=pt.get_default_dtype()))

        # Compute and append averaged within- and between-parcel correlations in current bin
        this_corr_within = pt.nanmean(cov[row[inBin[within]], col[inBin[within]]]) \
                           / pt.nanmean(var[row[inBin[within]], col[inBin[within]]])
        this_corr_between = pt.nanmean(cov[row[inBin[between]], col[inBin[between]]]) \
                            / pt.nanmean(var[row[inBin[between]], col[inBin[between]]])

        corr_within.append(this_corr_within)
        corr_between.append(this_corr_between)

        del inBin

    if weighting:
        weight = 1/(1/pt.stack(num_within) + 1/pt.stack(num_between))
        weight = weight / pt.sum(weight)
        DCBC = pt.nansum(pt.multiply((pt.stack(corr_within) - pt.stack(corr_between)), weight))
    else:
        DCBC = pt.nansum(pt.stack(corr_within) - pt.stack(corr_between))
        weight = pt.nan

    D = {
        "binWidth": binWidth,
        "maxDist": maxDist,
        "num_within": num_within,
        "num_between": num_between,
        "corr_within": corr_within,
        "corr_between": corr_between,
        "weight": weight,
        "DCBC": DCBC
    }

    return D

def get_parcel(atlas, parcel_name='MDTB10', do_plot=False):
    """Samples the existing MDTB10 parcellation
    Then displays it as check
    """
    atl_dir = base_dir + '/Atlases'
    with open(atl_dir + '/atlas_description.json') as file:
        atlases = json.load(file)
    if atlas not in atlases:
        raise(NameError(f'Unknown Atlas: {atlas}'))
    ainf = atlases[atlas]

    parcel = nb.load(atl_dir + '/%s/atl-%s_space-%s_dseg.nii'
                     % (ainf['dir'], parcel_name, ainf['space']))
    suit_atlas, _ = am.get_atlas(atlas, atl_dir)

    data = suit.reslice.sample_image(parcel,
            suit_atlas.world[0],
            suit_atlas.world[1],
            suit_atlas.world[2],0)

    # Read the parcellation colors: Add additional row for parcel 0
    ########################################################
    # The path of color .lut file to be changed if color info
    # stored in separate atlas folder. Right now, all colors are
    # stored in `tpl-SUIT` folder.
    ########################################################
    color_file = atl_dir + f'/tpl-SUIT/atl-{parcel_name}.lut'
    color_info = pd.read_csv(color_file, sep = ' ',header=None)
    colors = color_info.iloc[:,1:4].to_numpy()

    # Map Plot if requested (for a check)
    if do_plot:
        Nifti = suit_atlas.data_to_nifti(data)
        surf_data = suit.flatmap.vol_to_surf(Nifti,stats='mode')
        fig = suit.flatmap.plot(surf_data,render='plotly',
                                overlay_type='label',cmap=colors)
        fig.show()
    return data, colors

# def write_dlabel_cifti(parcellation, atlas, res='32k'):
#     #TODO: unfinished
#     if res == '32k':
#         VERTICES = 32492
#         bm_name = ['cortex_left', 'cortex_right']
#     else:
#         raise ValueError('Only fs_LR32k template is currently supported!')
#
#     if parcellation.dim() == 1:
#         # reshape to (1, num_vertices)
#         parcellation = parcellation.reshape(1,-1)
#
#     if parcellation.shape[1] == VERTICES*2:
#         # The input parcellation is already the full parcels
#         # (including medial wall)
#         pass
#     else:
#         # If the input parcellation is masked, we restore it
#         # to the full 32k vertices
#         par = np.full((1, VERTICES * 2), 0, dtype=int)
#         if atlas.structure == bm_name:
#             idx = np.hstack((atlas.vertex[0], atlas.vertex[1]+VERTICES))
#         if 'cortex_left' in stru.lower:
#             this_idx = atlas.vertex[i]
#             par[:, this_idx] =

def make_label_cifti(data,
                     anatomical_struct='Cerebellum',
                     labels=None,
                     label_names=None,
                     column_names=None,
                     label_RGBA=None):
    """Generates a label Cifti2Image from a numpy array

    Args:
        data (np.array):
             num_vert x num_col data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'Cerebellum'
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
        label_RGBA = np.zeros([num_labels,4])
        hsv = plt.cm.get_cmap('hsv',num_labels)
        color = hsv(np.linspace(0,1,num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        for i in range(num_labels):
            label_RGBA[i] = color[i]

    # Create label names from numerical values
    if label_names is None:
        label_names = []
        for i in labels:
            label_names.append("label-{:02d}".format(i))

    labelDict = [('???',(0,0,0,0))]
    for i, p in enumerate(Data):
        colorValue = (1, 1, 1, 1)
        if (p > 0):
            colorValue = colorMapping_p.to_rgba(p)
        elif (p < 0):
            colorValue = colorMapping_n.to_rgba(p)
        labelDict[p] = (i, colorValue)

    names = ['CIFTI_STRUCTURE_CORTEX_LEFT' for i in range(L_data.shape[0])]
    names.extend(['CIFTI_STRUCTURE_CORTEX_RIGHT' for i in range(R_data.shape[0])])
    verteces = [i for i in range(L_data.shape[0])]
    verteces.extend([i for i in range(L_data.shape[0])])
    verteces = np.asarray(verteces)
    brainModelAxis = nib.cifti2.cifti2_axes.BrainModelAxis(name=names, vertex=np.asarray(verteces),
                                                           nvertices={
                                                               'CIFTI_STRUCTURE_CORTEX_LEFT': 32492,
                                                               'CIFTI_STRUCTURE_CORTEX_RIGHT': 32492}, )
    newLabelAxis = nib.cifti2.cifti2_axes.LabelAxis(['aaa'], labelDict)
    newheader = nib.cifti2.cifti2.Cifti2Header.from_axes((newLabelAxis, brainModelAxis))
    newImage = nib.cifti2.cifti2.Cifti2Image(dataobj=Data.reshape([1, -1]), header=newheader)
    newImage.to_filename('%s/' % dir + name + '.dlabel.nii')


    # Create key-color mapping for labelAxis
    np.apply_along_axis(map(), 0, b)
    d = dict(enumerate(a))
    # Create label.gii structure
    C = nb.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    E_all = []
    for (label, rgba, name) in zip(labels, label_RGBA, label_names):
        E = nb.gifti.gifti.GiftiLabel()
        E.key = label
        E.label= name
        E.red = rgba[0]
        E.green = rgba[1]
        E.blue = rgba[2]
        E.alpha = rgba[3]
        E.rgba = rgba[:]
        E_all.append(E)

    D = list()
    for i in range(num_cols):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL',
            datatype='NIFTI_TYPE_UINT8',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.extend(E_all)
    return gifti