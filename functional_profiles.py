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
from ProbabilisticParcellation.scripts.parcel_hierarchy import analyze_parcel
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh, norm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from copy import deepcopy

def recover_info(info, model):
    """Recovers info fields that were lists from tsv-saved strings.
    Args:
        info: Model info loaded form tsv
    Returns:
        info: Model info with list fields.
        
    """
    # Recover model info from tsv file format
    for var in ['datasets', 'sess', 'type']:
        if not isinstance(info[var], list):
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
        profile: list of task profiles. Each entry is the V for one emission model
        conditions: list of conditions. Each entry is the condition for one emission model
        conditions_detailed: list of conditions. Each entry is the condition for one emission model along with the dataset name and the session name
    """
    # Get task profile for each emission model
    profile = [em.V for em in model.emissions]

    # Get conditions
    conds = []
    sessions = []
    for d,dname in enumerate(info.datasets):
        _, dinfo, dataset = get_dataset(base_dir,dname,atlas=info.atlas,sess=info.sess[d],type=info.type[d], info_only=True)

        cs = dinfo.drop_duplicates(subset=[dataset.cond_ind])
        cs = cs[dataset.cond_name].to_list()
        # remove whitespace and replace underscore with dash
        cs = [c.replace(' ', '').replace('_', '-')
                            for c in cs]
        conds.append(cs)
        sessions.append(dataset.sessions)
    
    allsessions = [ses for dset_sessions in sessions for ses in dset_sessions]
    
    conditions = []
    conditions_detailed = []
    
    if len(allsessions) == len(model.emissions):
        # If each session has an emission model, match conditions to profile vectors
        for d, dset_sessions in enumerate(sessions):
            for ses in dset_sessions:
                conditions_det = [f'{info.datasets[d][:2]}_{ses}_{con}' for con in conds[d]]
                conditions_srt = [
                    f'{con}' for con in conds[d]]
                conditions_detailed.append(conditions_det)
                conditions.append(conditions_srt)
                
    return profile, conditions, conditions_detailed

def show_parcel_profile(p, profiles, conditions, datasets, show_ds='all', ncond=5, print=True):
    """Returns the functional profile for a given parcel either for selected dataset or all datasets
    Args:
        profiles: parcel scores for each condition in each dataset
        conditions: condition names of each dataset
        ems: emission model descriptors (dataset names or datset + session combinations)
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

def get_profile_data(labels, info, profiles, conditions):
    sessions = [2, 1, 2, 14, 2, 1, 1]
    ems = []
    for d, dataset in enumerate(info.datasets):
        ems.extend([dataset] * sessions[d])


    label_profile = {}
    n_highest = 3
    df_dict = {}
    df_dict['dataset'] = []
    df_dict['label'] = []
    df_dict['n_label'] = []
    df_dict['conditions'] = []
    df_dict['parcel_no'] = []

    

    for l, label in enumerate(labels):
        if l != 0:

            parcel_no = labels.tolist().index(label) - 1
            profile = show_parcel_profile(
                parcel_no, profiles, conditions, ems, show_ds='all', ncond=1, print=False)
            highest_conditions = ['{}:{}'.format(ems[p][:2], ' & '.join(
                prof[:n_highest])) for p, prof in enumerate(profile)]
            label_profile[label] = highest_conditions

            for p in range(len(profile)):
                current_data = profile[p]
                for c in current_data:
                    df_dict['dataset'].append(ems[p])
                    df_dict['label'].append(label)
                    df_dict['n_label'].append(l)
                    df_dict['conditions'].append(c)
                    df_dict['parcel_no'].append(parcel_no)

    labels_alpha = sorted(label_profile.keys())
    df = pd.DataFrame(df_dict)
    return df


if __name__ == "__main__":
    # Merge C2 models
    space = 'MNISymC2'
    K=68
    mname = f'Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-{K}'
    info, model = load_batch_best(mname)
    info = recover_info(info, model)
    # for each parcel, get the highest scoring task
    profiles, conditions, cdetails = get_profiles(model=model, info=info)
    _, _, _, labels, _ = analyze_parcel(mname, sym=True)
    df = get_profile_data(labels, info, profiles, conditions)
    pass
    
