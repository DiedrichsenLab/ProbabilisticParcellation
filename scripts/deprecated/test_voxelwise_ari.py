# Test cases for ari
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import HierarchBayesParcel.evaluation as ev
import HierarchBayesParcel.emissions as em
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.evaluate as ppev
import matplotlib.pyplot as plt
import ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas as eva
import seaborn as sb
import sys
from util import *
import torch as pt
from sklearn.metrics import adjusted_rand_score


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
    corr = eva.compare_probs(a, b, method='corr')
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


def test_same_parcellation_ari(mname_A='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'):
    # load models
    info_a, model_a = ut.load_batch_best(mname_A)

    ari_voxelwise = ppev.ARI_voxelwise(pt.argmax(model_a.arrange.marginal_prob(), dim=0), pt.argmax(
        model_a.arrange.marginal_prob(), dim=0))

    print(
        f'Mean ARI_voxelwise: {ari_voxelwise.mean()}. All values are 1: {pt.all(ari_voxelwise == 1)}')
    return


def test_different_parcellation_ari(mname_A='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered', mname_B='Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem_reordered'):
    # load models
    # ppev.compare_voxelwise(mname_A, mname_B, individual=True)

    # prob_a = ppev.parcel_individual(
    #     mname_A, subject='all', dataset=None, session=None)

    # prob_b = ppev.parcel_individual(
    #     mname_B, subject='all', dataset=None, session=None)

    # pt.save(prob_a, f'{ut.model_dir}/Models/{mname_A}_Uhat.pt')
    # pt.save(prob_b, f'{ut.model_dir}/Models/{mname_B}_Uhat.pt')

    model_pair = ['Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered',
                  'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem_reordered']
    # load Uhats
    prob_a = pt.load(f'{ut.model_dir}/Models/{model_pair[0]}_Uhat.pt')
    prob_b = pt.load(f'{ut.model_dir}/Models/{model_pair[1]}_Uhat.pt')

    parcel_a = pt.argmax(prob_a, dim=1)
    parcel_b = pt.argmax(prob_b, dim=1)

    comparison, comparison_group = ppev.ARI_voxelwise(
        parcel_a, parcel_b).numpy()

    plt.imshow(prob_a[:, (prob_a.shape[1] // 2) -
               50: (prob_a.shape[1] // 2) + 50].squeeze())

    plt.imshow(prob_b[:, (prob_b.shape[1] // 2) -
                      50: (prob_b.shape[1] // 2) + 50].squeeze())

    # Make first 25 rows into list of 25 entries
    plt.figure(figsize=(20, 20))
    parcel_a_list = [parcel_a[i, :] for i in range(parcel_a.shape[0])]
    ax = ut.plot_multi_flat(parcel_a_list[:4], 'MNISymC2',
                            grid=(2, 2),
                            dtype='label',
                            colorbar=False,
                            cmap='tab20')
    ax.show()

    plt.figure(figsize=(20, 20))
    parcel_b_list = [parcel_b[i, :] for i in range(parcel_b.shape[0])]
    ax = ut.plot_multi_flat(parcel_b_list[:4], 'MNISymC2',
                            grid=(2, 2),
                            dtype='label',
                            colorbar=False,
                            cmap='tab20')
    ax.show()
    return


if __name__ == "__main__":
    # test_same_parcellation_ari()
    # test_voxelwise_probs()

    # test_ARI()
    # test_same_parcellation_ari()
    # test_corr()
    test_different_parcellation_ari()
    pass
