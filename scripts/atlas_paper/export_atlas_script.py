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
import nitools as nt

def export_atlas_gifti():
    model_names = [
        "Models_03/NettekovenSym68_space-MNISymC2",
        "Models_03/NettekovenAsym68_space-MNISymC2",
        "Models_03/NettekovenSym32_space-MNISymC2",
        "Models_03/NettekovenAsym32_space-MNISymC2",
    ]
    space='MNISymC2'
    for model_name in model_names:
        atlas_name = model_name.split("Models_03/")[1] 
        
        _, cmap, labels = nt.read_lut(ut.export_dir + f'{atlas_name.split("_space-")[0]}.lut')
        # add alpha value one to each rgb array
        cmap = np.hstack((cmap, np.ones((cmap.shape[0], 1))))
        
        # load model
        info, model = ut.load_batch_best(model_name)
        data = model.arrange.marginal_prob().numpy()

        ea.export_map(
            data,
            space,
            cmap,
            labels,
            f'{ut.model_dir}/Atlases/{atlas_name}',
        )



if __name__=="__main__":
    #ea.resample_atlas('NettekovenSym32','MNISymC2','MNI152NLin2009cSymC')
    #ea.resample_atlas('NettekovenAsym32','MNISymC2','MNI152NLin2009cSymC')
    #ea.resample_atlas('NettekovenSym68','MNISymC2','MNI152NLin2009cSymC')
    #ea.resample_atlas('NettekovenAsym68','MNISymC2','MNI152NLin2009cSymC')
    #ea.resample_atlas('NettekovenSym32','MNISymC2','SUIT')
    #ea.resample_atlas('NettekovenAsym32','MNISymC2','SUIT')
    #ea.resample_atlas('NettekovenSym68','MNISymC2','SUIT')
    #ea.resample_atlas('NettekovenAsym68','MNISymC2','SUIT')
    # ea.resample_atlas('NettekovenSym32','MNISymC2','MNI152NLin6AsymC')
    # ea.resample_atlas('NettekovenAsym32','MNISymC2','MNI152NLin6AsymC')
    # ea.resample_atlas('NettekovenSym68','MNISymC2','MNI152NLin6AsymC')
    # ea.resample_atlas('NettekovenAsym68','MNISymC2','MNI152NLin6AsymC')

    export_atlas_gifti()
    pass
