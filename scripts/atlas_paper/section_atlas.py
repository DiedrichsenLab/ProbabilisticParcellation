"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
import Functional_Fusion.dataset as ds
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
import ProbabilisticParcellation.scripts.atlas_paper.fit_C2_from_C3 as ft
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.evaluation as ev
import ProbabilisticParcellation.evaluate as ppev
import nibabel as nb

# from ProbabilisticParcellation.scripts.atlas_paper.parcel_hierarchy import analyze_parcel
import logging
import nitools as nt
import os

pt.set_default_tensor_type(pt.FloatTensor)

atlas_dir = (
    "/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/"
)


def dissect_atlas(atlname):
    '''Dissect the atlas into primary, secondary and tertiary parcels'''
    atlas_dir + atlname
    "NettekovenAsym32_space-MNISymC2_probseg"
    _, cmap, labels = nt.read_lut(atlas_dir + atlname.split("_space")[0] + ".lut")
    # Import probabilities and parcels
    Prob = nb.load(atlas_dir + atlname + "_probseg.nii")
    parcel = nb.load(atlas_dir + atlname + "_dseg.nii")
    


    pass



if __name__ == "__main__":
    dissect_atlas(atlname="NettekovenSym68_space-MNISymC2")
