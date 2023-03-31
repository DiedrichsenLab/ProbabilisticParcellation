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
from ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas import ARI_voxelwise, compare_probs
from sklearn.metrics import adjusted_rand_score


def test_same_parcellation_ari(mname_A='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'):
    # load models
    info_a, model_a = ut.load_batch_best(mname_A)

    ari_voxelwise = ARI_voxelwise(pt.argmax(model_a.arrange.marginal_prob(), dim=0), pt.argmax(
        model_a.arrange.marginal_prob(), dim=0))

    print(
        f'Mean ARI_voxelwise: {ari_voxelwise.mean()}. All values are 1: {pt.all(ari_voxelwise == 1)}')
    return


def test_different_parcellation_ari(mname_A='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered', mname_B='Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem_reordered'):
    # load models
    info_a, model_a = ut.load_batch_best(mname_A)
    _, model_b = ut.load_batch_best(mname_B)

    ari_voxelwise = ARI_voxelwise(pt.argmax(model_a.arrange.marginal_prob(), dim=0), pt.argmax(
        model_b.arrange.marginal_prob(), dim=0), individual=True)

    print(
        f'Mean ARI_voxelwise: {ari_voxelwise.mean()}. All values are 1: {pt.all(ari_voxelwise == 1)}')
    return


def test_voxelwise_probs():
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


def test_corr():
    # Initliaze tensor with two rows and random numbers
    a = pt.rand(4, 13)
    b = pt.rand(4, 13)
    corr = compare_probs(a, b, method='corr')
    a_folded = a[:2, :] + a[2:, :]
    b_folded = b[:2, :] + b[2:, :]
    corr_np = np.corrcoef(a_folded, b_folded)
    pass


def test_ARI():
    a = pt.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 1, 0, 1])
    b = pt.tensor([0, 0, 0, 1, 2, 1, 1, 2, 3, 1, 1, 0, 1])
    x = adjusted_rand_score(a, b)
    y = ev.ARI(a, b).item()
    pass


if __name__ == "__main__":
    # test_same_parcellation_ari()
    # test_voxelwise_probs()

    # test_ARI()
    # test_same_parcellation_ari()
    test_corr()
    pass
