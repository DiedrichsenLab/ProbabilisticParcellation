import numpy as np
import nibabel as nb
import SUITPy as suit
import pickle
from pathlib import Path
import Functional_Fusion.atlas_map as am
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt

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

def load_batch_best(fname):
    """ Loads a batch of model fits and selects the best one
    Args:
        fname (str): File name
    """
    info, models = load_batch_fit(fname)
    j = info.loglik.argmax()
    return info.iloc[j],models[j]

def extract_fit(models):
    """Extracts marginal probability values and V
    from a set of model fits 

    Args:
        models (list): List of FullMultiModel 
    """
    n_iter = len(models)
        # Intialize data arrays
    Prop = pt.zeros((n_iter,models[0].arrange.K,models[0].arrange.P))
    V = []

    for i,M in enumerate(models):
        Prop[i,:,:] = M.arrange.logpi.softmax(axis=0)

        # Now switch the emission models accordingly:
        for j,em in enumerate(M.emissions):
            if i==0:
                V.append(pt.zeros((n_iter,em.M,info.K[i])))
            V[j][i,:,:]=em.V
    return Prop,V

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
                    render='matplotlib'):
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
                overlay_type='label')
    # Plotting funtional data 
    elif dtype== 'func':
        surf_data = suit.flatmap.vol_to_surf(Nifti, stats='nanmean',
            space=map_space)
        ax = suit.flatmap.plot(surf_data, 
                render=render,
                cmap=cmap,
                cscale = cscale,
                new_figure=False,
                overlay_type='label')
    else:
        raise(NameError('Unknown data type'))
    return ax

def plot_multi_flat(data,atlas,grid,
                    cmap = None,
                    dtype = 'label',
                    cscale = None,
                    titles=None):
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
                    render='matplotlib') 
        if titles is not None: 
            plt.title(titles[i])

def plot_model_parcel(model_names,grid,cmap='tab20b'):
    """Load a bunch of model fits, selects the best from 
    each of them and plots the flatmap of the parcellation
    """
    parcel=[]
    titles = [] 
    # Extracts the data
    for i,mn in enumerate(model_names):
        info,model = load_batch_best(mn)
        Prop = model.marginal_prob()
        parcel.append(np.argmax(Prop,axis=0)+1) # Get winner take all

        # Split the name and build titles 
        split_mn = mn.split('_')
        titles.append(split_mn[1] + ' ' + split_mn[3])

    atlas = split_mn[2][6:]
    data = np.vstack(parcel)
    plot_multi_flat(data,atlas,grid=grid,
                     cmap=cmap,
                     titles=titles) 

if __name__ == "__main__":
    model_name = ['asym_Md_space-MNISymC3_K-10',
                  'asym_Po_space-MNISymC3_K-10',
                  'asym_Ni_space-MNISymC3_K-10']
    fig = plt.figure(figsize=(20,10))
    plot_model_parcel(model_name,[2,3])
    pass