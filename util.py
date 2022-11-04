import numpy as np
import nibabel as nb
import SUITPy as suit
import pickle
from pathlib import Path
import Functional_Fusion.atlas_map as am
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt
import generativeMRF.evaluation as ev

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))

def load_batch_fit(fname):
    """ Loads a batch of fits and extracts marginal probability maps 
    and mean vectors
    Args:
        fname (str): File name
    Returns: 
        info: Data Frame with information 
        models: List of models
    """
    wdir = base_dir + '/Models/'
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

def load_batch_best(fname):
    """ Loads a batch of model fits and selects the best one
    Args:
        fname (str): File name
    """
    info, models = load_batch_fit(fname)
    j = info.loglik.argmax()
    return info.iloc[j],models[j]

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
    suit_atlas = am.get_atlas(atlas,base_dir + '/Atlases')
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

    parc = np.argmax(Prob,axis=1)+1


    plot_multi_flat(parc,atlas,grid=grid,
                     cmap=cmap,
                     titles=titles) 
