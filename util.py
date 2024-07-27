import numpy as np
import SUITPy as suit
import pickle
import Functional_Fusion.atlas_map as am
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt
import HierarchBayesParcel.evaluation as ev
from pathlib import Path
import re

# Set directories for the entire project - just set here and import everywhere
# else
model_dir = 'Y:\data\Cerebellum\ProbabilisticParcellationModel'
home = str(Path.home())
if not Path(model_dir).exists():
    model_dir = '/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel'
if not Path(model_dir).exists():
    model_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(model_dir).exists():
    model_dir = '/Users/jdiedrichsen/Data/FunctionalFusion/'
if not Path(model_dir).exists():
    model_dir = str(Path(home, 'diedrichsen_data/data/Cerebellum/ProbabilisticParcellationModel'))
if not Path(model_dir).exists():
    raise (NameError('Could not find model_dir'))

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(base_dir).exists():
    base_dir = '/Users/jdiedrichsen/Data/FunctionalFusion/'
if not Path(base_dir).exists():
    base_dir = str(Path(home, 'diedrichsen_data/data/FunctionalFusion'))
if not Path(base_dir).exists():
    raise (NameError('Could not find base_dir'))
atlas_dir = base_dir + f'/Atlases'

figure_dir = "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/papers/AtlasPaper/Figure_parts/"
if not Path(figure_dir).exists():
    figure_dir = "/Users/callithrix/Dropbox/AtlasPaper/Figure_parts/"

export_dir = f'{base_dir}/../Cerebellum/ProbabilisticParcellationModel/Atlases/'
if not Path(export_dir).exists():
    export_dir = f'{base_dir}/Atlases/'

# pytorch cuda global flag
pt.set_default_dtype(pt.float32)
if pt.cuda.is_available():
    default_device = pt.device('cuda')
else:
    default_device = pt.device('cpu')

# Keep track of cuda memory


def report_cuda_memory():
    if pt.cuda.is_available():
        ma = pt.cuda.memory_allocated() / 1024 / 1024
        mma = pt.cuda.max_memory_allocated() / 1024 / 1024
        mr = pt.cuda.memory_reserved() / 1024 / 1024
        print(
            f'Allocated:{ma:.2f} MB, MaxAlloc:{mma:.2f} MB, Reserved {mr:.2f} MB')


def recover_info(info, model=None, mname=None, info_type='model_info'):
    """Recovers info fields that were lists from tsv-saved strings and adds model type information.
    Args:
        info: Model info loaded form tsv
    Returns:
        info: Model info with list fields.

    """
    if info_type == 'model_info':
        variables = ['datasets', 'sess', 'type']
        # Recover model info from tsv file format
        for var in variables:
            if not isinstance(info[var], list):
                v = eval(info[var])
                if len(model.emissions) > 2 and len(v) == 1:
                    v = eval(info[var].replace(" ", ","))
                info[var] = v

        model_settings = {
            "Models_01": [True, True, False],
            "Models_02": [False, True, False],
            "Models_03": [True, False, False],
            "Models_04": [False, False, False],
            "Models_05": [False, True, True],
        }

        info["model_type"] = f'Models_{mname.split("Models_")[1].split("/")[0]}'
        uniform_kappa = model_settings[info.model_type][0]
        joint_sessions = model_settings[info.model_type][1]

        info["uniform_kappa"] = uniform_kappa
        info["joint_sessions"] = joint_sessions
    elif info_type == 'evaluation_info':
        var = 'train_data'
        if not isinstance(info[var], list):
            v = eval(info[var])
            if len(v) == 1 and len(re.findall('[A-Z][^A-Z]*', v[0])) > 5:
                v = info[var].strip("[]'").split("' '")
            info[var] = v

    return info


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
    Y_tar = Y_target - Y_target.mean(dim=1, keepdim=True)
    Y_sou = Y_source - Y_source.mean(dim=1, keepdim=True)
    Cov = pt.matmul(Y_tar, Y_sou.t())
    Var1 = pt.sum(Y_tar * Y_tar, dim=1)
    Var2 = pt.sum(Y_sou * Y_sou, dim=1)
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
    info = pd.read_csv(wdir + fname + '.tsv', sep='\t')
    with open(wdir + fname + '.pickle', 'rb') as file:
        models = pickle.load(file)
    return info, models


def clear_batch(fname):
    """Ensures that pickle file does not contain superflous data
    Args:
        fname (): filename
    """
    wdir = base_dir + '/Models/'
    with open(wdir + fname + '.pickle', 'rb') as file:
        models = pickle.load(file)
    # Clear models
    for m in models:
        m.clear()

    with open(wdir + fname + '.pickle', 'wb') as file:
        pickle.dump(models, file)


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


def load_batch_best(fname, device=None):
    """ Loads a batch of model fits and selects the best one
    Args:
        fname (str): File name
    """
    info, models = load_batch_fit(fname)

    j = info.loglik.argmax()

    best_model = models[j]
    if device is not None:
        best_model.move_to(device)

    info_reduced = info.iloc[j]
    return info_reduced, best_model


def get_colormap_from_lut(fname=base_dir + '/Atlases/tpl-SUIT/atl-MDTB10.lut'):
    """ Makes a color map from a *.lut file
    Args:
        fname (str): Name of Lut file

    Returns:
        _type_: _description_
    """
    color_info = pd.read_csv(fname, sep=' ', header=None)
    color_map = np.zeros((color_info.shape[0] + 1, 3))
    color_map = color_info.iloc[:, 1:4].to_numpy()
    return color_map


def plot_data_flat(data, atlas,
                   cmap=None,
                   dtype='label',
                   cscale=None,
                   labels=None,
                   render='matplotlib',
                   colorbar=False,
                   bordersize=4,
                   bordercolor='k',
                   backgroundcolor='w'):
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
    suit_atlas, ainf = am.get_atlas(atlas, base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(data)

    # Mapping labels directly by the mode
    if dtype == 'label':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='mode',
                                             space=ainf['normspace'], ignore_zeros=True)
        ax = suit.flatmap.plot(surf_data,
                               render=render,
                               cmap=cmap,
                               new_figure=False,
                               label_names=labels,
                               overlay_type='label',
                               colorbar=colorbar,
                               bordersize=bordersize,
                               bordercolor=bordercolor,
                               backgroundcolor=backgroundcolor
                               )
    # Plotting one series of functional data
    elif dtype == 'func':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='nanmean',
                                             space=ainf['normspace'])
        surf_data = np.nan_to_num(surf_data)
        ax = suit.flatmap.plot(surf_data,
                               render=render,
                               cmap=cmap,
                               cscale=cscale,
                               new_figure=False,
                               overlay_type='func',
                               colorbar=colorbar,
                               bordersize=bordersize,
                               bordercolor=bordercolor,
                               backgroundcolor=backgroundcolor
                               )
    # Mapping probabilities on the flatmap and then
    # determining a winner from this (slightly better than label)
    elif dtype == 'prob':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='nanmean',
                                             space=ainf['normspace'])
        label = np.argmax(surf_data, axis=1) + 1
        ax = suit.flatmap.plot(label,
                               render=render,
                               cmap=cmap,
                               new_figure=False,
                               label_names=labels,
                               overlay_type='label',
                               colorbar=colorbar,
                               bordersize=bordersize,
                               bordercolor=bordercolor,
                               backgroundcolor=backgroundcolor
                               )
    else:
        raise (NameError('Unknown data type'))
    return ax


def plot_multi_flat(data, atlas, grid,
                    cmap=None,
                    dtype='label',
                    cscale=None,
                    titles=None,
                    colorbar=False,
                    save_fig=False,
                    save_under=None):
    """Plots a grid of flatmaps with some data

    Args:
        data (array or list): NxP array of data or list of NxP arrays of data (if plotting Probabilities)
        atlas (str): Atlas code ('SUIT3','MNISymC3',...)
        grid (tuple): (rows,cols) grid for subplot
        cmap (colormap or list): Color map or list of color maps. Defaults to None.
        dtype (str, optional):'label' or 'func'
        cscale (_type_, optional): Scale of data (None)
        titles (_type_, optional): _description_. Defaults to None.

    Returns:
        ax: Axis / figure of plot
    """
    if isinstance(data, np.ndarray):
        n_subplots = data.shape[0]
    elif isinstance(data, list):
        n_subplots = len(data)

    if not isinstance(cmap, list):
        cmap = [cmap] * n_subplots

    for i in np.arange(n_subplots):
        plt.subplot(grid[0], grid[1], i + 1)
        ax = plot_data_flat(data[i], atlas,
                            cmap=cmap[i],
                            dtype=dtype,
                            cscale=cscale,
                            render='matplotlib',
                            colorbar=(i == 0) & colorbar)
        if titles is not None:
            plt.title(titles[i])
            if save_fig:
                fname = f'rel_{titles[i]}.png'
                if save_under is not None:
                    fname = save_under
                plt.savefig(fname, format='png')
                # plt.savefig(f'rel_{titles[i]}_{i}.png', format='png',
                #             bbox_inches='tight', pad_inches=0)

    return ax


def hard_max(Prob):
    K, P = Prob.shape
    parcel = np.argmax(Prob, axis=0)
    U = np.zeros((K, P))
    U[parcel, np.arange(P)] = 1
    return U


def plot_model_pmaps(Prob, atlas, sym=True, labels=None, subset=None, grid=None):
    if isinstance(labels, list):
        labels = np.array(labels)
    K, P = Prob.shape
    if not sym:
        raise (NameError('only for symmetric models right now'))
    else:
        K = int(K / 2)
        PL = Prob[:K, :]
        PR = Prob[K:, :]
        Prob = PL + PR
        Prob[Prob > 1] = 1  # Exclude problems in the vermis
    if subset is None:
        subset = np.arange(K)
    if grid is None:
        a = int(np.ceil(np.sqrt(len(subset))))
        grid = (a, a)
    plot_multi_flat(Prob[subset, :], atlas, grid,
                    dtype='func',
                    cmap='Reds',
                    cscale=[0, 0.2],
                    titles=labels[subset],
                    colorbar=False,
                    save_fig=False)


def plot_connectivity_map(pscalar, surf, border, indx=0):
    if isinstance(pscalar, str):
        pscalar = np.load(pscalar)
    if isinstance(surf, str):
        pscalar = np.load(surf)

    pscalar = pscalar[indx, :]
    surf.plot.plotmap(
        DR, 'fs32k_R', underlay=s02sulc[1], cscale=[-3, 3], threshold=[-1, 1])


def plot_model_parcel(model_names, grid, cmap='tab20b', align=False, device=None):
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
    for i, mn in enumerate(model_names):
        info, model = load_batch_best(mn, device=device)
        models.append(model)
        # Split the name and build titles
        fname = mn.split('/')  # Get filename if directory is given
        split_mn = fname[-1].split('_')
        atlas = split_mn[2][6:]
        titles.append(split_mn[1] + ' ' + split_mn[3])

    # Align models if requested
    if align:
        Prob = ev.align_models(models, in_place=False)
    else:
        Prob = ev.extract_marginal_prob(models)

    if type(Prob) is pt.Tensor:
        if pt.cuda.is_available() or pt.backends.mps.is_built():
            Prob = Prob.cpu().numpy()
        else:
            Prob = Prob.numpy()

    parc = np.argmax(Prob, axis=1) + 1

    plot_multi_flat(Prob, atlas, grid=grid,
                    cmap=cmap, dtype='prob',
                    titles=titles)

    return Prob


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
        data = data - pt.mean(data, dim=1, keepdim=True)  # mean centering
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
    cov = pt.matmul(data, data.T) / (k - 1)
    data_sqrd = data**2
    data_sqrd[pt.isnan(data)] = 0  # Set NaN values to zero to ignore NaN
    sd = pt.sqrt((data_sqrd).sum(dim=1, keepdim=True) /
                 (k - 1))  # standard deviation
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
    The main DCBC calculation for volume space - same as in the DCBC package, but GPU accelerated
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
        inBin = pt.where((distance > i * binWidth) &
                         (distance <= (i + 1) * binWidth))[0]

        # lookup the row/col index of within and between vertices
        within = pt.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
        between = pt.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

        # retrieve and append the number of vertices for within/between in current bin
        num_within.append(
            pt.tensor(within.numel(), dtype=pt.get_default_dtype()))
        num_between.append(
            pt.tensor(between.numel(), dtype=pt.get_default_dtype()))

        # Compute and append averaged within- and between-parcel correlations in current bin
        this_corr_within = pt.nanmean(cov[row[inBin[within]], col[inBin[within]]]) \
            / pt.nanmean(var[row[inBin[within]], col[inBin[within]]])
        this_corr_between = pt.nanmean(cov[row[inBin[between]], col[inBin[between]]]) \
            / pt.nanmean(var[row[inBin[between]], col[inBin[between]]])

        corr_within.append(this_corr_within)
        corr_between.append(this_corr_between)

        del inBin

    if weighting:
        weight = 1 / (1 / pt.stack(num_within) + 1 / pt.stack(num_between))
        weight = weight / pt.sum(weight)
        DCBC = pt.nansum(pt.multiply(
            (pt.stack(corr_within) - pt.stack(corr_between)), weight))
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

if __name__ == "__main__":



    pass