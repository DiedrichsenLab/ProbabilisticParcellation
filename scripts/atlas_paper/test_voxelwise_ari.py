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


def test_same_parcellation_ari(mname_A='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'):
    # load models
    info_a, model_a = ut.load_batch_best(mname_A)

    ari_voxelwise = ARI_voxelwise(pt.argmax(model_a.arrange.marginal_prob(), dim=0), pt.argmax(
        model_a.arrange.marginal_prob(), dim=0))

    print(
        f'Mean ARI_voxelwise: {ari_voxelwise.mean()}. All values are 1: {pt.all(ari_voxelwise == 1)}')
    return


def test_voxelwise_probs(mname_A, mname_B, method='ari', save_nifti=False, plot=False, lim=None):
    mname1 = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'
    mname2 = 'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_reordered'

    # load models
    info_a, model_a = ut.load_batch_best(mname1)
    _, model_b = ut.load_batch_best(mname2)

    prob_a = model_a.arrange.marginal_prob()
    prob_b = model_b.arrange.marginal_prob()

    atlas = info_a.atlas
    suit_atlas, _ = am.get_atlas(atlas, ut.base_dir + '/Atlases')

    img_a = suit_atlas.data_to_nifti(prob_a)
    plt.imshow(img_a.get_fdata()[:, :, 20, 6].squeeze())

    img_b = suit_atlas.data_to_nifti(prob_b)
    plt.imshow(img_b.get_fdata()[:, :, 20, 6].squeeze())

    indx_left = np.where(suit_atlas.world[0, :] <= 0)[0]
    indx_right = np.where(suit_atlas.world[0, :] >= 0)[0]

    prob_a[:, indx_left] = 1
    img_a = suit_atlas.data_to_nifti(prob_a)
    plt.imshow(img_a.get_fdata()[:, :, 20, 41].squeeze())


if __name__ == "__main__":
    test_same_parcellation_ari()

    pass
