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
import Functional_Fusion.atlas_map as am
import SUITPy as suit
from PcmPy import matrix


# from ProbabilisticParcellation.scripts.atlas_paper.parcel_hierarchy import analyze_parcel
import logging
import nitools as nt
import os

pt.set_default_tensor_type(pt.FloatTensor)

atlas_dir = (
    "/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/"
)
suit_dir = "/Volumes/diedrichsen_data$/data/FunctionalFusion/Atlases/tpl-SUIT/"




def dissect_atlas(mname, from_model=True):
    '''Dissect the atlas into primary, secondary and tertiary parcels'''
    # Import probabilities and parcels
    fileparts = mname.split("/")
    space = mname.split("_space-")[1]
    atlasname = fileparts[1].split("_space")[0]

    info, model = ut.load_batch_best(mname)
    if from_model:
        Prob = model.marginal_prob().numpy()
        parcel = Prob.argmax(axis=0) + 1
    else:
        Prob = nb.load(atlas_dir + fileparts[1] + "_probseg.nii")
        parcel = nb.load(atlas_dir + fileparts[1] + "_dseg.nii")

    suit_atlas, _ = am.get_atlas(space, ut.base_dir + "/Atlases")
    Nifti = suit_atlas.data_to_nifti(parcel.astype(float))
    surf_parcel = suit.flatmap.vol_to_surf(Nifti, stats='mode',
                                        space='MNISymC', ignore_zeros=True)
    
    _, cmap, labels = nt.read_lut(atlas_dir + atlasname + ".lut")
    

    
    

    # Plot parcel size for left parcels
    D = cl.plot_parcel_size(Prob, ListedColormap(cmap), labels, wta=True, sort=False, side='L')
    plt.savefig(ut.figure_dir + f"parcel_sizes.pdf")
    
    # Get lobules 
    a, ainf = am.get_atlas(space, ut.atlas_dir)
    
    lobule_file =  "atl-Anatom_space-SUIT_dseg.nii"
    lobules = a.read_data(suit_dir + lobule_file, 0)

    # Get lobule labels
    _, _, lobule_labels = nt.read_lut(suit_dir + f"{lobule_file.split('_space-')[0]}.lut")

    # Primary section is in lobule 1 - 7a (superior to horizontal fissure)
    # Secondary section is in lobule 7a - 8 (between horizontal fissure and secondary fissure)
    # Tertiary section is in lobule 9 and 10 (inferior to secondary fissure)
    boundaries = ["Left_I_IV", "Left_CrusII", "Left_IX", "Left_Dentate"]
    # boundaries = ["Left_I_IV", "Right_CrusI", "Left_CrusII", "Right_VIIIb", "Left_IX", "Vermis_IX"]
    lobule_boundaries = [lobule_labels.index(b) for b in boundaries]
    # make empty mask with each section as a row
    mask = np.zeros((lobules.shape[0]))
    prim = (lobules > lobule_boundaries[0]) & (lobules <= lobule_boundaries[1])
    sec = (lobules > lobule_boundaries[1]) & (lobules <= lobule_boundaries[2])
    tert = (lobules > lobule_boundaries[2]) & (lobules <= lobule_boundaries[3])
    mask[prim] = 1
    mask[sec] = 2
    mask[tert] = 3
    
    # Plot mask via nifti
    # Nifti = suit_atlas.data_to_nifti(mask.astype(int))
    # lobules_surf = suit.flatmap.vol_to_surf(Nifti, stats='mode',
    #                                     space='MNISymC', ignore_zeros=True)
    # # get categorical colormap

    # suit.flatmap.plot(lobules_surf,
    #         render='matplotlib',
    #         cmap='Set1',
    #         new_figure=False,
    #         overlay_type='label')
    
    
    ut.plot_data_flat(mask.astype(int), 'MNISymC2',
                   dtype='label',
                   render='matplotlib',
                   bordersize=4,
                   cmap='Set1',
                   bordercolor='k',
                   backgroundcolor='w')
    plt.savefig(ut.figure_dir + f"lobule_sections.png")

    
    # Get indicator matrix for primary, secondary and tertiary parcels
    indicator_mask = matrix.indicator(mask)
    # Remove parcels where there is no lobule data (first column)
    indicator_mask = indicator_mask[:,1:]

    # Get sections by multiplying the indicator matrix with the Prob matrix, so that I have the probabilities for each parcel split into sections
    # Multiply a matrix with a vector without summing over the columns
    prim_prob = indicator_mask[:,0].T * Prob
    sec_prob = indicator_mask[:,1].T * Prob
    tert_prob = indicator_mask[:,2].T * Prob
    # Zip the sections into one matrix
    zip_sections = zip(prim_prob, sec_prob, tert_prob)
    Prob_sections = np.vstack(zip_sections)
    cmap_sections = np.vstack((cmap[0], np.repeat(cmap[1:], 3, axis=0)))

    # Sort parcels that start with M labels first, then A, D, S
    domains = ['M', 'A', 'D', 'S']
    regions = np.arange(1,5)
    sections = ['superior', 'dorsal', 'inferior']
    hemispheres = ['L', 'R']
    labels_sections = ['0'] + [f"{d}{r}_{s}_{h}" for h in hemispheres for d in domains  for r in regions  for s in sections]

    
    
    
    # Plot parcel size for left parcels
    D = cl.plot_parcel_size(Prob_sections, ListedColormap(cmap_sections), labels_sections, wta=True, sort=False, side='L')
    # Adjust figure size
    plt.gcf().set_size_inches(10, 40)
    plt.savefig(ut.figure_dir + f"parcel_sizes_SDI.pdf")

    
    
    
    
    

    



    # secion_primary = 
    # section_secondary =
    # section_tertiary =  



    pass



if __name__ == "__main__":
    dissect_atlas(mname="Models_03/NettekovenSym32_space-MNISymC2")

