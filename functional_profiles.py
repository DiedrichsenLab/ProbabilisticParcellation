""" Analyze functional profiles across emsssion models
"""

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
import PcmPy as pcm
import nibabel as nb
import nibabel.processing as ns
import SUITPy as suit
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import pickle
from ProbabilisticParcellation.util import *
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh, norm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from copy import deepcopy

def recover_info(info):
    """Recovers info fields that were lists from tsv-saved strings.
    Args:
        info: Model info loaded form tsv
    Returns:
        info: Model info with list fields.
        
    """
    # Recover model info from tsv file format
    for var in ['datasets', 'sess', 'type']:
        v = eval(info[var])
        if len(model.emissions) > 2 and len(v) == 1:
            v = eval(info[var].replace(' ', ','))
        info[var] = v
    return info

def get_profiles(model, info):
    """Returns the functional profile for each parcel
    Args:
        model: Loaded model
        info: Model info
    Returns:
        profile: V for each emission model
        conditions: list of condition lists for each dataset
    """
    # Get task profile for each emission model
    profile = [em.V for em in model.emissions]

    # Get conditions
    conditions = []
    for d,dname in enumerate(info.datasets):
        _, dinfo, dataset = get_dataset(base_dir,dname,atlas=info.atlas,sess=info.sess[d],type=info.type[d], info_only=True)

        condition_names = dinfo.drop_duplicates(subset=[dataset.cond_ind])
        condition_names = condition_names[dataset.cond_name].to_list()
        conditions.append([condition.split('  ')[0] for condition in condition_names])

    # Match profile vectors to conditions
    # (Several emission models for the same dataset)
    conditions
    return profile, conditions

def show_parcel_profile(p, profiles, conditions, datasets, show_ds='all', ncond=5, print=True):
    """Returns the functional profile for a given parcel either for selected dataset or all datasets
    Args:
        profiles: parcel scores for each condition in each dataset
        conditions: condition names of each dataset
        datasets: dataset names
        show_ds: selected dataset
                'Mdtb'
                'Pontine'
                'Nishimoto'
                'Ibc'
                'Hcp'
                'all'
        ncond: number of highest scoring conditions to show

    Returns:
        profile: condition names in order of parcel score

    """
    if show_ds =='all':
        # Collect condition names in order of parcel score from all datasets
        profile = []
        for d,dataset in enumerate(datasets):
            cond_name = conditions[d]
            cond_score = profiles[d][:,p].tolist()
            # sort conditions by condition score
            dataset_profile = [name for _,name in sorted(zip(cond_score,cond_name),reverse=True)]
            profile.append(dataset_profile)
            if print:
                print('{} :\t{}'.format(dataset, dataset_profile[:ncond]))

    else:
        # Collect condition names in order of parcel score from selected dataset
        d = datasets.index(show_ds)
        cond_name = conditions[d]
        cond_score = profiles[d][:,p].tolist()

        # sort conditions by condition score
        dataset_profile = [name for _,name in sorted(zip(cond_score,cond_name))]
        profile = dataset_profile
        if print:
            print('{} :\t{}'.format(datasets[d], dataset_profile[:ncond]))

    return profile


if __name__ == "__main__":
    # Merge C2 models
    space = 'MNISymC2'
    K=68
    mname = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-{K}'
    info, model = load_batch_best(mname)
    info = recover_info(info)
    # for each parcel, get the highest scoring task
    profiles, conditions = get_profiles(model=model, info=info)

    pass
    
