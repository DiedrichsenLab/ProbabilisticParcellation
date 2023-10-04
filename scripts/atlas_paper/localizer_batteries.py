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

# pytorch cuda global flag
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)

# import condition info
dataset = ds.get_dataset_class(ut.base_dir, 'Mdtb')
dinfo = dataset.get_info(ses_id='ses-s1', type='CondRun')


localizer_language = [
    'UnpleasantScenes',
    'PleasantScenes',
    'SadFaces',
    'HappyFaces',
    'ToM',
    'VideoAct',
    'FingerSeq',
    'rest']

localizer_demand = [
    'Motor imagery',
    'ToM',
    'VideoAct',
    'FingerSeq',
    'SpatialNavigation',
    'VerbGen',
    'WordRead',
    'rest']

localizer_social = [
    'NoGo',
    'Math',
    'IntervalTiming',
    'StroopIncon',
    'FingerSeq',
    'Object2Back',
    'VisualSearchLarge',
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

# create a dictionary of localizer batteries
localizer_batteries = {'language':localizer_language,
                       'demand':localizer_demand,
                       'social':localizer_social,
                       'general':localizer_general}
    


if __name__ == "__main__":
    mname = 'Models_03/NettekovenSym32_space-MNISymC2'
    info,model = ut.load_batch_best(mname,device=ut.default_device)
    for key, localizer_tasks in localizer_batteries.items():
        # Make localizer task strings into regressor indicators
        # get cond_num_uni where localizer_tasks entry equals cond_name
        localizer_tasks = [dinfo['cond_num_uni'][dinfo['cond_name']==task].values[0] for task in localizer_tasks]
        
        # _, Uhat_complete, _ = ig.get_individ_group_mdtb(model,atlas='MNISymC2',  localizer_tasks=localizer_tasks)
        Uhat_data, Uhat_complete, Uhat_group = ig.get_individ_group_mdtb(model,atlas='MNISymC2',  localizer_tasks=localizer_tasks)
        D = ig.evaluate_dcbc(Uhat_data, Uhat_complete, Uhat_group,atlas='MNISymC2',max_dist=40)
        D['localizer'] = key

        # Export
        fname = ut.model_dir+ f'/Models/Evaluation_03/localizer_battery_{key}_sym32.tsv'
        D.to_csv(fname,sep='\t')
        # Concatenate dataframes
        if key == 'language':
            D_all = D
        else:
            D_all = pd.concat([D_all,D])
        pass
    fname = ut.model_dir+ '/Models/Evaluation_03/localizer_battery_sym32.tsv'
    D_all.to_csv(fname,sep='\t')
    pass