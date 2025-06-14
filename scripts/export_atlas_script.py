"""
Minimal script to export the atlas to different spaces
"""

import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from ProbabilisticParcellation.util import export_dir, model_dir, load_batch_best
import ProbabilisticParcellation.export_atlas as ea
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

        _, cmap, labels = nt.read_lut(export_dir + f'{atlas_name.split("_space-")[0]}.lut')
        # add alpha value one to each rgb array
        cmap = np.hstack((cmap, np.ones((cmap.shape[0], 1))))

        # load model
        info, model = load_batch_best(model_name)
        data = model.arrange.marginal_prob().numpy()

        ea.export_map(
            data,
            space,
            cmap,
            labels,
            f'{model_dir}/Atlases/{atlas_name}',
        )

def resample_atlas_all():
    ea.resample_atlas('NettekovenSym32','MNISymC2','MNI152NLin2009cSymC')
    ea.resample_atlas('NettekovenAsym32','MNISymC2','MNI152NLin2009cSymC')
    ea.resample_atlas('NettekovenSym68','MNISymC2','MNI152NLin2009cSymC')
    ea.resample_atlas('NettekovenAsym68','MNISymC2','MNI152NLin2009cSymC')
    ea.resample_atlas('NettekovenSym32','MNISymC2','SUIT')
    ea.resample_atlas('NettekovenAsym32','MNISymC2','SUIT')
    ea.resample_atlas('NettekovenSym68','MNISymC2','SUIT')
    ea.resample_atlas('NettekovenAsym68','MNISymC2','SUIT')
    ea.resample_atlas('NettekovenSym32','MNISymC2','MNI152NLin6AsymC')
    ea.resample_atlas('NettekovenAsym32','MNISymC2','MNI152NLin6AsymC')
    ea.resample_atlas('NettekovenSym68','MNISymC2','MNI152NLin6AsymC')
    ea.resample_atlas('NettekovenAsym68','MNISymC2','MNI152NLin6AsymC')


def subdivide_spatial_all():
    ea.subdivde_atlas_spatial(fname='NettekovenSym32',atlas='SUIT',outname='NettekovenSym128')
    ea.subdivde_atlas_spatial(fname='NettekovenAsym32',atlas='SUIT',outname='NettekovenAsym128')
    ea.subdivde_atlas_spatial(fname='NettekovenSym32',atlas='MNI152NLin2009cSymC',outname='NettekovenSym128')
    ea.subdivde_atlas_spatial(fname='NettekovenAsym32',atlas='MNI152NLin2009cSymC',outname='NettekovenAsym128')
    ea.subdivde_atlas_spatial(fname='NettekovenSym32',atlas='MNI152NLin6AsymC',outname='NettekovenSym128')
    ea.subdivde_atlas_spatial(fname='NettekovenAsym32',atlas='MNI152NLin6AsymC',outname='NettekovenAsym128')

    """After running this function, copy the atlas files into the cerebellar_atlases repo, gzip the probseg files (for easier storage) and push to cerbellar_atlases github repo"""
   

    pass


if __name__=="__main__":
    # resample_atlas_all()
    # subdivide_spatial_all()
    export_atlas_gifti()
    pass
