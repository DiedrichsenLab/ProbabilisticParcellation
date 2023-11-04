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

def get_order(mname, reorder):
    """Returns the ordering index for the atlas regions
    Args:
        reorder (str, optional): [description]. Defaults to 'first_clustering'.
        Options: 
            - first_clustering: First clustering of the parcels according to functional similarity
            - introspection_to_action: Merges standalone introspection region (lobule IX region) into action 
            - tongue: Swaps A1 and M2, because A1 was mislabelled as action region, when it actually was tongue region
            - action4_to_social5: Moves A4 into social cluster and renames it to S5, since it is involved in scene reconstruction / imagination

    """
    
    replace=None

    if reorder=="first_clustering":
        # For swapping M2 and A1
        if "32" in mname :
            original_idx = "parcel_orig_idx"
            mname_new = mname + "_firstClustering"
        elif "68" in mname :
            # For swapping M2 and A1 in K68
            original_idx = "XX"
            mname_new = mname + "_firstClustering"

    elif reorder=="introspection_to_action":
        # For swapping M2 and A1
        if "32" in mname :
            original_idx = "parcel_orig_idx_5Domains"
            mname_new = mname + "_4Domains"
        elif "68" in mname :
            # For swapping M2 and A1 in K68
            original_idx = "parcel_med_idx_5Domains"
            mname_new = mname + "_4Domains"
    
    elif reorder=="tongue":
        # For swapping M2 and A1
        if "32" in mname :
            original_idx = "parcel_med_before_tongue_swap_idx"
            mname_new = mname + "_tongueSwap"
        elif "68" in mname :
            # For swapping M2 and A1 in K68
            original_idx = "parcel_orig_idx_before_tongue_swap_idx"
            mname_new = mname + "_tongueSwap"
    elif reorder=="action4_to_social5":
        if "32" in mname :
            original_idx = "parcel_med_before_a4_swap_idx"
            mname_new = mname + "_a4Swap"
            replace = {
            "A4L": "S5L",
            "A4R": "S5R"
            }
        elif "68" in mname :
            # For swapping M2 and A1 in K68
            original_idx = "parcel_orig_idx_before_a4_swap_idx"
            mname_new = mname + "_a4Swap"
            replace = {
                "A4La": "S5La",
                "A4Ra": "S5Ra"
            }

    return original_idx, mname_new, replace
            

def reorder_atlas(reorder='action4_to_social5'):
    """Reorders the atlas based on the functional similarity of the parcels
    Args:
        reorder (str, optional): [description]. Defaults to 'first_clustering'.
        Options: 
            - first_clustering: First clustering of the parcels according to functional similarity
            - introspection_to_action: Merges standalone introspection region (lobule IX region) into action 
            - tongue: Swaps A1 and M2, because A1 was mislabelled as action region, when it actually was tongue region
            - action4_to_social5: Moves A4 into social cluster and renames it to S5, since it is involved in scene reconstruction / imagination

        """

    mnames = [
        "Models_03/NettekovenSym68_space-MNISymC2",
        "Models_03/NettekovenAsym68_space-MNISymC2",
        "Models_03/NettekovenSym32_space-MNISymC2",
        "Models_03/NettekovenAsym32_space-MNISymC2",
    ]
    f_assignment = 'mixed_assignment_68_32_4.csv'

    for mname in mnames:

        original_idx, mname_new, replace = get_order(mname, reorder)
        
        symmetry = mname.split("/")[1].split("_")[0]
        if symmetry == "sym":
            sym = True
        else:
            sym = False

        model_reordered = ea.reorder_model(
            mname,
            sym=sym,
            assignment=f_assignment,
            original_idx=original_idx,
            save_model=True,
            mname_new = mname_new
        )


def reorder_lut(reorder='action4_to_social5'):
    """Reorders the atlas based on the functional similarity of the parcels
    Args:
        reorder (str, optional): [description]. Defaults to 'first_clustering'.
        Options: 
            - first_clustering: First clustering of the parcels according to functional similarity
            - introspection_to_action: Merges standalone introspection region (lobule IX region) into action 
            - tongue: Swaps A1 and M2, because A1 was mislabelled as action region, when it actually was tongue region
            - action4_to_social5: Moves A4 into social cluster and renames it to S5, since it is involved in scene reconstruction / imagination

        """

    mnames = [
        "Models_03/NettekovenSym68_space-MNISymC2",
        "Models_03/NettekovenAsym68_space-MNISymC2",
        "Models_03/NettekovenSym32_space-MNISymC2",
        "Models_03/NettekovenAsym32_space-MNISymC2",
    ]
    f_assignment = 'mixed_assignment_68_32_4.csv'

    for mname in mnames:

        original_idx, mname_new, replace = get_order(mname, reorder)
        # Make new lut file
        index, colors, labels = nt.read_lut(ut.export_dir + f'archive/atl-{mname.split("Models_03/")[1].split("_space-")[0]}.lut')
        assignment = pd.read_csv(f"{ut.model_dir}/Atlases/{f_assignment}")

        order_index = assignment[original_idx].values
        new_order = np.unique(order_index, return_index=True)[1]
        new_order=np.array([order_index[index] for index in sorted(new_order)])
        new_order = np.concatenate([new_order, new_order + len(new_order)])

        # Make new labels
        new_labels = [labels[1:][i] for i in new_order]
        if replace:
            for k,v in replace.items():
                new_labels[new_labels.index(k)] = v
        new_labels = ["0"] + new_labels
        
        # Make new colours
        new_colors = colors[1:][new_order]
        new_colors = np.vstack((np.ones((new_colors.shape[1])), new_colors))

        # Make new index
        new_index = np.arange(len(new_labels))

        # Save new lut file
        nt.save_lut(ut.export_dir + f'{mname.split("Models_03/")[1].split("_space-")[0]}.lut',new_index,new_colors,new_labels)

    # Reorder domain lut file
    index, colors, labels = nt.read_lut(ut.export_dir + f'archive/NettekovenSym32_domain.lut')
    domain_labels = [labels[1:][i] for i in new_order]
    if replace:
        for k,v in replace.items():
            domain_labels[domain_labels.index(k[:2])] = v[:2]
    domain_labels = ["0"] + domain_labels

    domain_colors = colors[1:][new_order]
    domain_colors = np.vstack((np.ones((domain_colors.shape[1])), domain_colors))


        



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

def subdivide_spatial_all():
    ea.subdivde_atlas_spatial(fname='NettekovenSym32',atlas='SUIT',outname='NettekovenSym128')
    ea.subdivde_atlas_spatial(fname='NettekovenAsym32',atlas='SUIT',outname='NettekovenAsym128')
    ea.subdivde_atlas_spatial(fname='NettekovenSym32',atlas='MNI152NLin2009cSymC',outname='NettekovenSym128')
    ea.subdivde_atlas_spatial(fname='NettekovenAsym32',atlas='MNI152NLin2009cSymC',outname='NettekovenAsym128')
    ea.subdivde_atlas_spatial(fname='NettekovenSym32',atlas='MNI152NLin6AsymC',outname='NettekovenSym128')
    ea.subdivde_atlas_spatial(fname='NettekovenAsym32',atlas='MNI152NLin6AsymC',outname='NettekovenAsym128')

if __name__=="__main__":
    # reorder_atlas()
    reorder_lut()
    # ea.resample_atlas('NettekovenSym32','MNISymC2','MNI152NLin2009cSymC')
    # ea.resample_atlas('NettekovenAsym32','MNISymC2','MNI152NLin2009cSymC')
    # ea.resample_atlas('NettekovenSym68','MNISymC2','MNI152NLin2009cSymC')
    # ea.resample_atlas('NettekovenAsym68','MNISymC2','MNI152NLin2009cSymC')
    # ea.resample_atlas('NettekovenSym32','MNISymC2','SUIT')
    # ea.resample_atlas('NettekovenAsym32','MNISymC2','SUIT')
    # ea.resample_atlas('NettekovenSym68','MNISymC2','SUIT')
    # ea.resample_atlas('NettekovenAsym68','MNISymC2','SUIT')
    # ea.resample_atlas('NettekovenSym32','MNISymC2','MNI152NLin6AsymC')
    # ea.resample_atlas('NettekovenAsym32','MNISymC2','MNI152NLin6AsymC')
    # ea.resample_atlas('NettekovenSym68','MNISymC2','MNI152NLin6AsymC')
    # ea.resample_atlas('NettekovenAsym68','MNISymC2','MNI152NLin6AsymC')
    # subdivide_spatial_all()
    # export_atlas_gifti()
    pass
