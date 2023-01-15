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
from matplotlib import pyplot as plt
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.learn_fusion_gpu as lf

def make_c2_model(mname, to_atlas = 'MNISymC2'):
    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    info,model = load_batch_best(mname)
    atlas = am.get_atlas(to_atlas)
    tic = time.perf_counter()
    data, cond_vec, part_vec, subj_ind = build_data_list(info.datasets,
                                                         atlas=atlas.name,
                                                         sess=info.sess,
                                                         type=info.type,
                                                         join_sess=False)
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    # Load all necessary data and designs
    n_sets = len(data)

    print(f'Building fullMultiModel {arrange} + {emission} for fitting...')
    M = build_model(arrange,sym_type,emission,atlas,weighting)


    pass

if __name__ == "__main__":
    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'
    m = make_c2_model(mname)
