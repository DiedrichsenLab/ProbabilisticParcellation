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



def reorder_lut(version=3, mnames_new=["NettekovenSym"]):
    """Reorders the lut file based on the new order of the parcels.
    Colour map was only created for version 1 onwards, therefore version 0 is not supported.
    Creates a lut file for the finest granularity (68 parcels), the medium granularity (32 parcels) and the coarsest granularity (4 domains).

    Args:
        version (str, optional): Defaults to 1.
        Options: 
            - v0: 5 domain model
            - v1: 4 domain model
            - v2: 4 domain model with correct Tongue region
            - v3: 4 domain model with correct Tongue region and S5

        mnames_new (list, optional): Names of the new models.
        """

    # Get assignment and v0 lut 
    assignment = pd.read_csv(f"{ut.model_dir}/Atlases/assignment.csv")
    original_lut = f'{ut.export_dir}/original.lut'
    index, colors, labels = nt.read_lut(original_lut)
    
    # --- K68 ---
    # Reorder colors for K68
    order_arrange = assignment[f"idx_v{version}"]
    new_colors = np.vstack((np.zeros((colors.shape[1])), colors[1:][order_arrange], colors[1:][order_arrange]))

    # Get new labels for K68
    new_labels = np.array(assignment[f'labels_v{version}']).tolist()
    new_labels = ["0"] + [label[:2] + 'L' + label[2] for label in new_labels] + [label[:2] + 'R' + label[2] for label in new_labels]
    # Save
    nt.save_lut(f'{ut.export_dir}/{mname.split("Models_03/")[1].split("_space-")[0]}.lut', index, new_colors, new_labels)

    # --- K32 ---
    # New labels for K32
    new_labels = np.array([label[:2] for label in new_labels])
    new_labels = new_labels[np.sort(np.unique(new_labels, return_index=True)[1])]
    new_labels = ['0'] + [label + 'L' for label in new_labels[1:]] + [label + 'R' for label in new_labels[1:]]
    # New colors for K32
    new_colors = new_colors[np.sort(np.unique(new_labels, return_index=True)[1])]
    new_colors = np.vstack((np.zeros((colors.shape[1])), new_colors[1:]))
    # New index for K32
    new_index = np.arange(len(new_labels))
    # Save
    nt.save_lut(f'{ut.export_dir}/{mname.split("Models_03/")[1].split("_space-")[0][:-2]}32.lut', new_index, new_colors, new_labels)

    # --- K4 ---
    # Make lut file with domain colours
    domain_colours = {'M': [0.4, 0.71, 0.98], 'A': [0.3239, 0.3067, 0.881], 'D': [0.8, 0.3261, 0.7], 'S': [0.98, 0.76, 0.13], 'I': [0.3724, 1.0, 1.0]}
    # Note that introspection colour is not used past version 0 because introspection regions are integrated into other domains
    domain_colours = np.vstack((np.zeros((colors.shape[1])), np.array([domain_colours[label[0]] for label in new_labels[1:]])))

    nt.save_lut(f'{ut.export_dir}/{mname.split("Models_03/")[1].split("_space-")[0]}_domain.lut', new_index, domain_colours, new_labels)





def reorder_models(version=3, mnames=["Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68", "Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem"]):
    """Creates a model version from the model selected to be the base map.
        The parcels are reordered according to the specified model version.
        Every model with 32 regions is created by first merging parcels of the 68 parcel model v0 into 32 parcels and refitting the emission models before reordering to obtain the desired model version.

        The following model versions can be created:
            - v0: 5 domain model
            - v1: 4 domain model
            - v2: 4 domain model with correct Tongue region
            - v3: 4 domain model with correct Tongue region and S5

        N.B.:
            - v0: First reordering of the parcels according to functional similarity (fine level) and clusters into five domains (medium level)
            - v1: Moves standalone introspection regions I1 & I2 up into action domain (fine level), creating a 4 domain model (medium level)
            - v2: Swaps A1 and M2, because A1 was mislabelled as action region, when it actually was tongue region and vice versa.
            - v3: Moves A4 into social cluster and renames it to S5, since it is involved in scene reconstruction / imagination
        
        To create each model, all previous reordering steps are applied. That means, to create v3, v0, v1 and v2 reordering steps are applied in this order.

        Args:
            version (int, optional): Defaults to 3.
            models (list, optional): List of model names that serve as the base model for creating all model versions.
    
    """

    for mname in mnames:
        
        
        info, model = ut.load_batch_best(mname)
        sym=~("asym" in mname)
        fileparts = mname.split("/")
        symmetry = fileparts[1].split("_")[0]
        symmetry = symmetry[0].upper() + symmetry[1:]
        space = mname.split("_space-")[1].split("_")[0]

        # reorder model to create fine granularity
        mname_new = f"Models_03/test_Nettekoven{symmetry}68_space-{space}"
        model_reordered, new_info = cl.reorder_model(mname, model, info, sym, version=version, mname_new=mname_new, save_model=True)        

        # merge model to create medium granularity 
        info = new_info.iloc[0]
        mname_new = f"{fileparts[0]}/test_Nettekoven{symmetry}32_space-{space}"
        model, info, mname_new, labels = cl.cluster_parcel(
            mname, model_reordered, new_info.iloc[0], mname_new=mname_new, version=version, refit_model=True, save_model=True
        )



        reorder_lut(version=version, mname_new=f'Nettekoven{symmetry}')

        
    
    pass

if __name__ == "__main__":
    base_models = [
        "Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68",
        "Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem"]
    version = 3
    reorder_models(version=version, mnames=base_models)
    
    
