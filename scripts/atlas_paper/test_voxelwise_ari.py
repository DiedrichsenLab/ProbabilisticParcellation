# Test cases for ari
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import generativeMRF.evaluation as ev
import generativeMRF.emissions as em
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.evaluate as ppev
import matplotlib.pyplot as plt
import seaborn as sb
import sys
from util import *
import torch as pt
from ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas import ARI_voxelwise
from sklearn.metrics import adjusted_rand_score

def test_same_parcellation_ari(mname_A='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'):
    # load models
    info_a, model_a = ut.load_batch_best(mname_A)

    ari_voxelwise = ARI_voxelwise(pt.argmax(model_a.arrange.marginal_prob(), dim=0), pt.argmax(
        model_a.arrange.marginal_prob(), dim=0))

    print(
        f'Mean ARI_voxelwise: {ari_voxelwise.mean()}. All values are 1: {pt.all(ari_voxelwise == 1)}')
    return


def test_ARI(): 
    a = pt.tensor([1,2,3,1,2,3,1,2,3,0,1,0,1])
    b = pt.tensor([0,0,0,1,2,1,1,2,3,1,1,0,1])
    x=adjusted_rand_score(a,b)
    y=ev.ARI(a,b).item()
    pass


if __name__ == "__main__":
    test_ARI()
    # test_same_parcellation_ari()
    pass
