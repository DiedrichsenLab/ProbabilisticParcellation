# Evaluate cerebellar parcellations
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import HierarchBayesParcel.evaluation as ev
import HierarchBayesParcel.emissions as em

import ProbabilisticParcellation.evaluate as ppev
import matplotlib.pyplot as plt
import seaborn as sb
import sys
from util import *
import torch as pt


base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))

pt.set_default_tensor_type(pt.DoubleTensor)


def noise_floor_dep_N(K=10,iter=20):
    P=1000
    T=pd.DataFrame()
    for N in [5,8,10,15,20,25,30,100]:
        cos_err = []        
        for n in range(iter):
            emission = em.MixVMF(K,N,P)
            emission.random_params()
            tdata = pt.normal(0,1,(10,N,P))
            emission.initialize(tdata)
            U = pt.softmax(emission.Estep(tdata), dim=1)

            eval= ev.coserr(tdata, emission.V, U,
                        adjusted=True, soft_assign=True)
            cos_err.append(eval.mean().item())
        D = pd.DataFrame({'N':[N]*iter,'cos_err':cos_err})
        T=pd.concat([T,D],ignore_index=True)
    return T


def run_prederr():
    model = 'Models_02_archive/asym_Md_space-MNISymC3_K-10'
    R = ppev.run_prederror(model,'Somatotopic','all',
        cond_ind=None,
        part_ind='half',
        eval_types=['group','floor'],
        indivtrain_ind=None,indivtrain_values=[1])
    pass


if __name__ == "__main__":
    T = noise_floor_dep_N()
    pass
   
   