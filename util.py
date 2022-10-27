import numpy as np
import nibabel as nb
import SUITPy as suit
import pickle
from pathlib import Path

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))


def load_batch_best(fname):
    """ Loads a batch of model fits and selects the best one
    Args:
        fname (str): File name
    """
    wdir = base_dir + '/Models/'
    info = pd.read_csv(wdir + fname + '.tsv',sep='\t')
    with open(wdir + fname + '.pickle','rb') as file:
        models = pickle.load(file)
    j = info.loglik.argmax()
    return info.iloc[j],models[j]

def plot_parcel_flat(parcel,cmap,atlas,render='matplotlib'):
    # Plot Parcellation 
    suit_atlas = am.get_atlas(atlas,base_dir + '/Atlases')
    Nifti = suit_atlas.data_to_nifti(parcel)
    if atlas[0:4]=='SUIT':
        map_space='SUIT'
    elif atlas[0:7]=='MNISymC':
        map_space='MNISymC'
    else:
        raise(NameError('Unknown atlas space'))

    surf_data = suit.flatmap.vol_to_surf(Nifti, stats='mode',
            space=map_space,ignore_zeros=True)
    fig = suit.flatmap.plot(surf_data, 
                render=render,
                cmap=cmap, 
                new_figure=True,
                overlay_type='label')
    return fig
