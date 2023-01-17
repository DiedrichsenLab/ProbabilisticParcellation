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

def make_highres_model(info,model, new_space = 'MNISymC2'):
    atlas,ainf = am.get_atlas(new_space,atlas_dir)
    split_mn = info['name'].split('_')

    tic = time.perf_counter()
    ds = eval(info.datasets.replace(' ',','))
    se = eval(info.sess.replace(' ',','))
    ty = eval(info.type.replace(' ',','))
    data, cond_vec, part_vec, subj_ind = lf.build_data_list(ds,
                                                         atlas=atlas.name,
                                                         sess=se,
                                                         type=ty,
                                                         join_sess=False)
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    # Load all necessary data and designs
    n_sets = len(data)

    print(f'Building fullMultiModel {info.arrange} + {info.emission} for fitting...')
    M = lf.build_model(model.K,info.arrange,split_mn[0],info.emission,atlas,
            cond_vec,part_vec,
            model.emissions[0].uniform_kappa)
    ## Copy over parameters
    
    M.initialize(data, subj_ind=subj_ind)
    return M

def refit_model_in_new_space(mname,new_space='MNISymC2'):
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info,model = load_batch_best(mname)
    M = make_highres_model(info,model,new_space)
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
    info = pd.DataFrame(info.to_dict(),index=[0])
    info['loglik']=ll3[-1]
    info['atlas']=new_space
    wdir = model_dir + f'/Models/' + fileparts[-2]
    fname = f'/{split_mn[0]}_{split_mn[1]}_space-{new_space}_K-{M.K}'
    info.to_csv(wdir + fname + '.tsv', sep='\t',index=False)
    with open(wdir + fname + '.pickle', 'wb') as file:
            pickle.dump(M, file)


if __name__ == "__main__":
    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-60'
    refit_model_in_new_space(mname,new_space='MNISymC2')