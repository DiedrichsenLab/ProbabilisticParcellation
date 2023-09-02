"""
Minimal script to export the atlas to different spaces
"""

import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from scipy.linalg import block_diag
import torch as pt
import matplotlib.pyplot as plt
import torch as pt
import time
from matplotlib import pyplot as plt
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.learn_fusion_gpu as lf

if __name__=="__main__":
    ea.resample_atlas('NettekovenSym32','MNISymC2','MNI152NLin2009cSymC')
    ea.resample_atlas('NettekovenSym32','MNISymC2','SUIT')
    ea.resample_atlas('NettekovenAsym32','MNISymC2','MNI152NLin2009cSymC')
    ea.resample_atlas('NettekovenAsym32','MNISymC2','SUIT')
l    pass
