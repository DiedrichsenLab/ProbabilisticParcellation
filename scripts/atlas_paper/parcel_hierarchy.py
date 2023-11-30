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
import HierarchBayesParcel.evaluation as ev
import nitools as nt

pt.set_default_tensor_type(pt.FloatTensor)



def reorder_lut(version=3, mname_new=["NettekovenSym68"]):
    """Reorders the lut file based on the new order of the parcels.
    Colour map was only created for version 1 onwards, therefore version 0 is not supported.

    Args:
        version (str, optional): Defaults to 1.
        Options: 
            - v1: 4 domain model
            - v2: 4 domain model with correct Tongue region
            - v3: 4 domain model with correct Tongue region and S5

        mname_new (list, optional): Names of the new models.
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


def reorder_models(version=3, mnames=["Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68", "Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem"]):
    """Creates a model version from the model selected to be the base map.
        The parcels are reordered according to the specified model version.
        Every model with 32 regions is created by first merging parcels of the 68 parcel model v0 into 32 parcels and refitting the emission models before reordering to obtain the desired model version.

        The following model versions can be created:
            - v0: First reordering of the parcels according to functional similarity (fine level) and clusters into five domains (medium level)
            - v1: Moves standalone introspection regions I1 & I2 up into action domain (fine level), creating a 4 domain model (medium level)
            - v2: Swaps A1 and M2, because A1 was mislabelled as action region, when it actually was tongue region and vice versa.
            - v3: Moves A4 into social cluster and renames it to S5, since it is involved in scene reconstruction / imagination
        
        To create each model, all previous reordering steps are applied. That means, to create v3, v0, v1 and v2 reordering steps are applied in this order.

        Args:
            version (int, optional): Defaults to 3.
            models (list, optional): List of model names that serve as the base model for creating all model versions.
    
    """

    mnames_new = []
    for mname in mnames:
        
        # reorder model to create fine granularity
        info, model = ut.load_batch_best(mname)
        sym=~("asym" in mname)
        fileparts = mname.split("/")
        symmetry = fileparts[1].split("_")[0]
        symmetry = symmetry[0].upper() + symmetry[1:]
        space = mname.split("_space-")[1].split("_")[0]
        K = info['K']
        # mname_new = f"{fileparts[0]}/Nettekoven{symmetry}{K}_space-{space}"
        mname_new = f"{fileparts[0]}/test_Nettekoven{symmetry}{K}_space-{space}"
        new_model, new_info = cl.reorder_model(mname, model, info, sym, version=version, mname_new=mname_new, save_model=True)        
        mnames_new.append(mname_new)

        # merge model to create medium granularity 
        info = new_info.iloc[0]
        K = info['K']
        # mname_new = f"{fileparts[0]}/Nettekoven{symmetry}{K}_space-{space}"
        mname_new = f"{fileparts[0]}/test_Nettekoven{symmetry}{K}_space-{space}"
        model, info, mname_new, labels = cl.cluster_parcel(
            mname, new_model, new_info.iloc[0], mname_new=mname_new, version=version, refit_model=True, save_model=True
        )
        mnames_new.append(mname_new)



    reorder_lut(version=version, mname_new=mnames_new)

        
    
    pass




if __name__ == "__main__":
    base_models = [
        "Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68",
        "Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem"]
    version = 3
    reorder_models(version=version, mnames=base_models)
    
    
