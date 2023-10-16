# Script for comparing predictive performance of different localizer batteries
import pandas as pd
from pathlib import Path
import numpy as np
import torch as pt
import nibabel as nb
import SUITPy as suit
import matplotlib.pyplot as plt
import seaborn as sb
from copy import copy,deepcopy
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.matrix as matrix
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.evaluation as ev
import ProbabilisticParcellation.util as ut 
import ProbabilisticParcellation.evaluate as ppev
import ProbabilisticParcellation.individ_group as ig
import nitools as nt

# pytorch cuda global flag
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# import condition info
dataset = ds.get_dataset_class(ut.base_dir, 'Mdtb')
dinfo = dataset.get_info(ses_id='ses-s1', type='CondRun')


localizer_language = [
    'MotorImagery',
    'ToM',
    'VideoAct',
    'FingerSeq',
    'SpatialNavigation',
    'VerbGen',
    'WordRead',
    'rest']

localizer_demand = [
    'NoGo',
    'Math',
    'IntervalTiming',
    'StroopIncon',
    'FingerSeq',
    'Object2Back',
    'VisualSearchLarge',
    'rest']

localizer_social = [
    'UnpleasantScenes',
    'PleasantScenes',
    'SadFaces',
    'HappyFaces',
    'ToM',
    'VideoAct',
    'FingerSeq',
    'rest']

localizer_general = [
    'ToM',
    'VideoAct',
    'FingerSeq',
    'Object2Back',
    'VisualSearchLarge',
    'SpatialNavigation',
    'VerbGen',
    'rest']

localizer_other = [
    'Go',
    'DigitJudgement',
    'ObjectViewing',
    'Verbal2Back',
    'ToM',
    'VideoAct',
    'FingerSeq',
    'rest']


# create a dictionary of localizer batteries
# localizer_batteries = {'language':localizer_language,
#                        'demand':localizer_demand,
#                        'social':localizer_social,
#                        'general':localizer_general}

localizer_batteries = {'language':localizer_language,
                       'demand':localizer_demand,
                       'social':localizer_social,
                       'general':localizer_general,
                       'other':localizer_other}


def get_masks(atlas, model):
    _, cmap, labels = nt.read_lut(f'{ut.export_dir}{atlas}.lut') 
    pseg = model.arrange.marginal_prob().numpy()
    dseg = np.argmax(pseg, axis=0) + 1
    
    M_labels = [l for l,label in enumerate(labels) if label[0]=='M']
    mask_motor = np.isin(dseg,M_labels)

    A_labels = [l for l,label in enumerate(labels) if label[0]=='A']
    mask_action = np.isin(dseg,A_labels)

    S_labels = [l for l,label in enumerate(labels) if label[0]=='S']
    mask_socioling = np.isin(dseg,S_labels)

    D_labels = [l for l,label in enumerate(labels) if label[0]=='D']
    mask_demand = np.isin(dseg,D_labels)

    return mask_motor, mask_action, mask_socioling, mask_demand

if __name__ == "__main__":
    atlas = 'NettekovenSym32'
    space = 'MNISymC2'
    max_dist = 40
    mname = f'Models_03/{atlas}_space-{space}'
    info,model = ut.load_batch_best(mname,device=ut.default_device)
    
    mot, act, soc, dem = get_masks(atlas,model)
    socdem = np.logical_or(soc,dem)
    masks = {
            'motor':mot,
            'action':act,
            'socio-ling':soc,
            'demand':dem,
            'socio-ling_demand':socdem}


    # --- Sanity check: Plot mask ---
    # Nifti = suit_atlas.data_to_nifti(np.array([int(val) for val in mask_demand_socioling]))
    # surf_data = suit.flatmap.vol_to_surf(Nifti, stats='mode',
    #                                  space='MNISymC', ignore_zeros=False)
    # ax = suit.flatmap.plot(surf_data,
    #                    render='matplotlib',
    #                    new_figure=True,
    #                    label_names=labels,
    #                    overlay_type='label',
    #                    colorbar=False,
    #                    bordersize=0)

    Data = []
    for key, localizer_tasks in localizer_batteries.items():
        # Make localizer task strings into regressor indicators
        localizer_tasks = [dinfo['cond_num_uni'][dinfo['cond_name']==task].values[0] for task in localizer_tasks]

        # Get Uhat from localizer scans
        Uhat_data, Uhat_complete, Uhat_group = ig.get_individ_group_mdtb(model,atlas=space,  localizer_tasks=localizer_tasks)

        # Calculate DCBC across entire cerebellum
        D = ig.evaluate_dcbc(Uhat_data, Uhat_complete, Uhat_group,atlas=space,max_dist=max_dist)
        D['localizer'] = key
        D['mask'] = 'whole cerebellum'
        data = [D]
        # data = []
        
        # Calculate DCBC within masks
        for mask_name, mask in masks.items():
            D = ig.evaluate_dcbc(Uhat_data, Uhat_complete, Uhat_group,atlas=space,max_dist=max_dist, mask=mask)
            D['localizer'] = key
            D['mask'] = mask_name
            data.append(D)

        # Export
        data = pd.concat(data)
        fname = ut.model_dir+ f'/Models/Evaluation_03/localizer_battery_{key}_sym32_spatial.tsv'
        data.to_csv(fname,sep='\t')

        # Append to master dataframe
        Data.append(data)


    fname = ut.model_dir+ '/Models/Evaluation_03/localizer_battery_sym32_spatial.tsv'
    Data = pd.concat(Data)
    Data.to_csv(fname,sep='\t')
    pass