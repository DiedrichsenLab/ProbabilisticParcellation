"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
from Functional_Fusion.dataset import *
from scipy.linalg import block_diag
import torch as pt
import matplotlib.pyplot as plt
import ProbabilisticParcellation.util as ut
import PcmPy as pcm
import torch as pt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from copy import deepcopy
import ProbabilisticParcellation.learn_fusion_gpu as lf
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.functional_profiles as fp
import Functional_Fusion.dataset as ds
import generativeMRF.evaluation as ev
import logging

pt.set_default_tensor_type(pt.FloatTensor)


def correlate_profile(data, profile):
    pass


def get_cortex(method='corr', mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'):
    space = mname.split('space-')[1].split('_')[0]
    info, model = ut.load_batch_best(mname)
    info = fp.recover_info(info, model, mname)
    profile_file = f'{ut.model_dir}/Atlases/{mname.split("/")[-1]}_task_profile_data.tsv'
    if Path(profile_file).exists():
        profile = pd.read_csv(
            profile_file, sep="\t"
        )
    else:

        parcel_profiles, profile_data = fp.get_profile(model, info)
    data = []
    for dset in info.datasets:
        d, i, dataset = ds.get_dataset(
            ut.base_dir, dset, atlas=space, sess='all')
        data.append(d)

    cortex = correlate_profile(data, parcel_profiles)
    return cortex


if __name__ == "__main__":

    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'
    cortex = get_cortex(mname=mname, method='corr')
    pass
