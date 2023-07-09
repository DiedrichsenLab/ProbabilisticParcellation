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

def size_plot(Prob, cmap, labels):
    # Plot parcel size for left parcels
    if 'superior' in labels[1]:
        plt.figure(figsize=(10, 40))
        D = cl.plot_parcel_size(Prob, ListedColormap(cmap), labels, wta=True, sort=False, side='L')
        plt.savefig(ut.figure_dir + f"parcel_sizes_section.pdf")
        
        plt.figure(figsize=(10, 40))
        D = cl.plot_parcel_size(Prob, ListedColormap(cmap), labels, wta=False, sort=False, side='L')
        plt.savefig(ut.figure_dir + f"parcel_sizes_section_probs.pdf")
        
        if 'vermis' in labels[4]:
            plt.figure(figsize=(10, 40))
            D = cl.plot_parcel_size(Prob, ListedColormap(cmap), labels, wta=True, sort=False, side='L')
            plt.savefig(ut.figure_dir + f"parcel_sizes_section_vermis.pdf")
            
            plt.figure(figsize=(10, 40))
            D = cl.plot_parcel_size(Prob, ListedColormap(cmap), labels, wta=False, sort=False, side='L')
            plt.savefig(ut.figure_dir + f"parcel_sizes_section_probs_vermis.pdf")

    else:
        plt.figure()
        D = cl.plot_parcel_size(Prob, ListedColormap(cmap), labels, wta=True, sort=False, side='L')
        plt.savefig(ut.figure_dir + f"parcel_sizes.pdf")

        plt.figure()
        D = cl.plot_parcel_size(Prob, ListedColormap(cmap), labels, wta=False, sort=False, side='L')
        plt.savefig(ut.figure_dir + f"parcel_sizes_probs.pdf")

def import_lobules(space):
    # Get lobules 
    a, ainf = am.get_atlas(space, ut.atlas_dir)
    
    lobule_file =  "atl-Anatom_space-SUIT_dseg.nii"
    lobules = a.read_data(suit_dir + lobule_file, 0)

    # Get lobule labels
    _, _, lobule_labels = nt.read_lut(suit_dir + f"{lobule_file.split('_space-')[0]}.lut")
    return lobules, lobule_labels


def get_sections(lobules, lobule_labels, vermis=False):
    # Primary section is in lobule 1 - 7a (superior to horizontal fissure)
    # Secondary section is in lobule 7a - 8 (between horizontal fissure and secondary fissure)
    # Tertiary section is in lobule 9 and 10 (inferior to secondary fissure)

    boundaries = ["Left_I_IV", "Left_CrusII", "Left_IX", "Left_Dentate"]
    lobule_boundaries = [lobule_labels.index(b) for b in boundaries]

    # make empty mask with each section as a row
    mask = np.zeros((lobules.shape[0]))
    prim = (lobules > lobule_boundaries[0]) & (lobules <= lobule_boundaries[1])
    sec = (lobules > lobule_boundaries[1]) & (lobules <= lobule_boundaries[2])
    tert = (lobules > lobule_boundaries[2]) & (lobules <= lobule_boundaries[3])
    mask[prim] = 1
    mask[sec] = 2
    mask[tert] = 3

    if vermis:
        # Section out the vermis
        vermis_indices = [idx+1 for idx, label in enumerate(lobule_labels) if 'Vermis' in label]  
        prim_vermis = prim & (np.isin(lobules, vermis_indices))
        sec_vermis = sec & (np.isin(lobules, vermis_indices))
        tert_vermis = tert & (np.isin(lobules, vermis_indices))
        mask[prim_vermis] = 4
        mask[sec_vermis] = 5
        mask[tert_vermis] = 6
    
    # Plot mask   
    plt.figure()
    ut.plot_data_flat(mask.astype(int), 'MNISymC2',
                   dtype='label',
                   render='matplotlib',
                   bordersize=4,
                   cmap='Set1',
                   bordercolor='k',
                   backgroundcolor='w')
    if vermis:
        plt.savefig(ut.figure_dir + f"lobule_sections_vermis.png")
    else:
        plt.savefig(ut.figure_dir + f"lobule_sections.png")

    # Get indicator matrix for primary, secondary and tertiary parcels
    indicator_mask = matrix.indicator(mask)
    # Remove parcels where there is no lobule data (first column)
    indicator_mask = indicator_mask[:,1:]
    
    return mask, indicator_mask


def divide_parcels(Prob, indicator_mask, cmap, vermis=False):
    # Get sections by multiplying the indicator matrix with the Prob matrix, so that I have the probabilities for each parcel split into sections
    # Multiply a matrix with a vector without summing over the columns
    prim_prob = indicator_mask[:,0].T * Prob
    sec_prob = indicator_mask[:,1].T * Prob
    tert_prob = indicator_mask[:,2].T * Prob
    sections = ['superior', 'dorsal', 'inferior']
    if vermis:
        sections.extend(['vermis_superior', 'vermis_dorsal', 'vermis_inferior'])
        prim_verm_prob = indicator_mask[:,3].T * Prob
        sec_verm_prob = indicator_mask[:,4].T * Prob
        tert_verm_prob = indicator_mask[:,5].T * Prob
        zip_sections = zip(prim_prob, sec_prob, tert_prob, prim_verm_prob, sec_verm_prob, tert_verm_prob)
        n_sections = 6
    else:
        zip_sections = zip(prim_prob, sec_prob, tert_prob)
        n_sections = 3
    Prob_sections = np.vstack(zip_sections)
    cmap_sections = np.vstack((cmap[0], np.repeat(cmap[1:], n_sections, axis=0)))

    # Sort parcels that start with M labels first, then A, D, S
    domains = ['M', 'A', 'D', 'S']
    regions = np.arange(1,5)
    hemispheres = ['L', 'R']
    labels_sections = ['0'] + [f"{d}{r}_{s}_{h}" for h in hemispheres for d in domains  for r in regions  for s in sections]

    return Prob_sections, cmap_sections, labels_sections

def dissect_atlas(mname):
    '''Dissect the atlas into primary, secondary and tertiary parcels'''
    # --- Import probabilities and parcels ---
    fileparts = mname.split("/")
    space = mname.split("_space-")[1]
    atlasname = fileparts[1].split("_space")[0]

    info, model = ut.load_batch_best(mname)
    Prob = model.marginal_prob().numpy()
    parcel = Prob.argmax(axis=0) + 1

    suit_atlas, _ = am.get_atlas(space, ut.base_dir + "/Atlases")
    Nifti = suit_atlas.data_to_nifti(parcel.astype(float))
    surf_parcel = suit.flatmap.vol_to_surf(Nifti, stats='mode',
                                        space='MNISymC', ignore_zeros=True)
    
    _, cmap, labels = nt.read_lut(atlas_dir + atlasname + ".lut")
    
    size_plot(Prob, cmap, labels)

    # --- Get lobules ---
    lobules, lobule_labels = import_lobules(space)  

    # --- Get sections ---
    _, indicator_mask = get_sections(lobules, lobule_labels, vermis=True)

    
    # --- Divide parcels into sections ---
    Prob_sections, cmap_sections, labels_sections = divide_parcels(Prob, indicator_mask, cmap, vermis=True)
    
    # --- Plot section sizes ---
    size_plot(Prob_sections, cmap_sections, labels_sections)




if __name__ == "__main__":
    dissect_atlas(mname="Models_03/NettekovenSym32_space-MNISymC2")

