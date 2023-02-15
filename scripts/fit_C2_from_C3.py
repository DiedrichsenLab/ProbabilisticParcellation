"""
Fits the 
"""

import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from scipy.linalg import block_diag
import torch as pt
import matplotlib.pyplot as plt
from ProbabilisticParcellation.util import *
import torch as pt
import time
from matplotlib import pyplot as plt
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.learn_fusion_gpu as lf


def make_highres_model(info, model, new_space='MNISymC2'):
    atlas, ainf = am.get_atlas(new_space, atlas_dir)
    split_mn = info['name'].split('_')

    tic = time.perf_counter()
    info_lists = {}
    for var in ['datasets', 'sess', 'type']:
        v = eval(info[var])
        if len(v) < len(split_mn[1]) / 2:
            v = eval(info[var].replace(' ', ','))
        info_lists[var] = v

    data, cond_vec, part_vec, subj_ind = lf.build_data_list(info_lists['datasets'],
                                                            atlas=atlas.name,
                                                            sess=info_lists['sess'],
                                                            type=info_lists['type'],
                                                            join_sess=False)
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    # Load all necessary data and designs
    n_sets = len(data)

    print(
        f'Building fullMultiModel {info.arrange} + {info.emission} for fitting...')
    M = lf.build_model(model.K, info.arrange, split_mn[0], info.emission, atlas,
                       cond_vec, part_vec,
                       model.emissions[0].uniform_kappa)
    # Copy over parameters
    for i, em in enumerate(model.emissions):
        for param in em.param_list:
            setattr(M.emissions[i], param, getattr(em, param))
    M.initialize(data, subj_ind=subj_ind)
    return M


def refit_model_in_new_space(mname, new_space='MNISymC2'):
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info, model = load_batch_best(mname)
    M = make_highres_model(info, model, new_space)
    M.move_to(device=default_device)
    M, ll1, theta, Uhat = M.fit_em(iter=100, tol=0.01,
                                   fit_emission=False,
                                   fit_arrangement=True,
                                   first_evidence=True)
    M, ll2, theta, Uhat = M.fit_em(iter=100, tol=0.01,
                                   fit_emission=True,
                                   fit_arrangement=False,
                                   first_evidence=True)
    M, ll3, theta, Uhat = M.fit_em(iter=100, tol=0.01,
                                   fit_emission=True,
                                   fit_arrangement=True,
                                   first_evidence=True)

    # make info from a Series back to a dataframe
    info = pd.DataFrame(info.to_dict(), index=[0])
    info['loglik'] = ll3[-1].item()
    info['atlas'] = new_space
    wdir = model_dir + f'/Models/' + fileparts[-2]
    fname = f'/{split_mn[0]}_{split_mn[1]}_space-{new_space}_K-{M.K}'
    info.to_csv(wdir + fname + '.tsv', sep='\t', index=False)
    with open(wdir + fname + '.pickle', 'wb') as file:
        pickle.dump([M], file)


if __name__ == "__main__":

    target_space = 'MNISymC2'

    # ks = [68, 80]
    ks = [32]

    # ks = [56]
    for k in ks:
        mname = f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-{k}'
        refit_model_in_new_space(mname, new_space=target_space)

    # move models that are still on cuda to cpu
#     for k in ks:
#         mname = f'Models_03/sym_MdPoNiIbWmDeSo_space-{target_space}_K-{k}'
#         inf, m = load_batch_fit(mname)
#         if m[0].ds_weight.is_cuda:
#             print(f'Convert model with K={k} {mname} to cpu...')
#             lf.move_batch_to_device(mname, device='cpu')
