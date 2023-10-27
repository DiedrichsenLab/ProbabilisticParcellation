#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for learning fusion on datasets

Created on 02/15/2023 at 2:16 PM
Author: cnettekoven
"""
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.learn_fusion_gpu as lf
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import Functional_Fusion.matrix as matrix
import nibabel as nb
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.evaluation as ev
import torch as pt
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import time
import glob


if __name__ == "__main__":
    files = sorted(
        glob.glob(f'{ut.model_dir}//Models/Evaluation/nettekoven_68/eval_*'))

    for f, file in enumerate(files):
        t = pd.read_csv(file, delimiter='\t')
        if f == 0:
            D = pd.DataFrame()
        if not 'source' in t.columns:
            t['source'] = 'sym_' + \
                file.split('/')[-1].strip('.tsv').split('_sym_')[-1]
            if '\n' in t.train_data[0]:
                t.train_data = t.train_data.apply(lambda x: x.split('\n')[-2])
            t.to_csv(file, sep='\t', index=False)
