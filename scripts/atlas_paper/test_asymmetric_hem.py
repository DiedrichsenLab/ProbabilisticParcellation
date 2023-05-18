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
import generativeMRF.arrangements as ar
import ProbabilisticParcellation.learn_fusion_gpu as lf


def test_asym_sym(mname_sym='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68', mname_asym='Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'):
    # load models
    info_sym, model_sym = ut.load_batch_best(mname_sym)
    info_sym = ut.recover_info(info_sym, model_sym, mname_sym)

    info_asym, model_asym = ut.load_batch_best(mname_asym)
    info_asym = ut.recover_info(info_asym, model_asym, mname_asym)

    atlas, _ = am.get_atlas(info_sym.atlas, ut.atlas_dir)
    indx_hem = np.sign(atlas.world[0, :])

    model_settings = {'Models_01': [True, True, False],
                      'Models_02': [False, True, False],
                      'Models_03': [True, False, False],
                      'Models_04': [False, False, False],
                      'Models_05': [False, True, True]}

    # uniform_kappa = model_settings[new_info.model_type][0]
    join_sess = model_settings[info_sym.model_type][1]
    join_sess_part = model_settings[info_sym.model_type][2]

    datasets = info_sym.datasets
    sessions = info_sym.sess
    types = info_sym.type

    data, cond_vec, part_vec, subj_ind = lf.build_data_list(datasets,
                                                            atlas=info_sym.atlas,
                                                            sess=sessions,
                                                            type=types,
                                                            join_sess=join_sess,
                                                            join_sess_part=join_sess_part)

    # Test symmetric arrangement model
    M = fm.FullMultiModel(model_sym.arrange, model_sym.emissions)

    # Attach the data
    M.initialize(data, subj_ind=subj_ind)
    M.fit_em(iter=1, tol=0.01,
             fit_emission=True,
             fit_arrangement=True,
             first_evidence=True)
    M.marginal_prob()
    M.arrange.map_to_full()

    plt.imshow(Umap[0, :, :].squeeze())
    plt.imshow(ARItest[:, (ARItest.shape[1] // 2) -
               50: (ARItest.shape[1] // 2) + 50].squeeze())
    plt.imshow(Umap[0, :, :].squeeze())

    # model_sym.arrange.map_to_full()

    # Old way of fitting asymmetric model
    # ar_model = ar.ArrangeIndependentSymmetric(K,
    #                                           atlas.indx_full,
    #                                           atlas.indx_reduced,
    #                                           same_parcels=False,
    #                                           spatial_specific=True,
    #                                           remove_redundancy=False)

    # Make arrangement model asymmetric but with hemispheres fitted separately
    new_arrange = ar.ArrangeIndependentSeparateHem(model_sym.K,
                                                   indx_hem=indx_hem,
                                                   indx_full=model_sym.arrange.indx_full,
                                                   spatial_specific=model_sym.arrange.spatial_specific,
                                                   remove_redundancy=model_sym.arrange.rem_red,
                                                   )

    # Test Estep, marginal_prob(), and map_to_full()

    new_arrange.arrange.map_to_arrange()
    return


if __name__ == "__main__":

    test_asym_sym()
    pass
