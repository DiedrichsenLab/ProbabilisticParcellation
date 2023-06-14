"""
Script to analyze individual variability
"""

import pandas as pd
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import Functional_Fusion.dataset as ds
import ProbabilisticParcellation.util as ut
from copy import deepcopy
import ProbabilisticParcellation.learn_fusion_gpu as lf
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.functional_profiles as fp
import ProbabilisticParcellation.scripts.atlas_paper.symmetry as sm
import ProbabilisticParcellation.scripts.atlas_paper.describe_atlas as da
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.evaluation as ev
import Functional_Fusion.atlas_map as am
import SUITPy as suit
import os
import nitools as nt

pt.set_default_tensor_type(pt.FloatTensor)


figure_path = (
    "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/papers/AtlasPaper/figure_parts/"
)
if not os.path.exists(figure_path):
    figure_path = "/Users/callithrix/Dropbox/AtlasPaper/figure_parts/"
atlas_dir = (
    "/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/"
)


def inter_individual_variability(probs_indiv, subject_wise=False):
    """Get inter-individual variability between individual parcellations by correlating probablistic parcellations voxelwise"""
    Corr = []
    for vox in np.arange(probs_indiv.shape[2]):
        corr = np.corrcoef(probs_indiv[:, :, vox])
        if subject_wise:
            # Set diagonal to nan
            np.fill_diagonal(corr, np.nan)
            # Retun average per row
            corr = np.nanmean(corr, axis=1)
            # Append to list
            Corr.append(corr)
        else:
            corr = corr[np.triu_indices(corr.shape[0], k=1)]
            Corr.append(np.nanmean(corr))

    return np.array(Corr)


def reliability_norm(corr, dataset, subject_wise=False):
    """Normalize inter-individual variability by intra-individual reliability"""
    reliability_voxelwise, _ = ds.reliability_maps(
        ut.base_dir,
        dataset,
        atlas="MNISymC2",
        subtract_mean=True,
        voxel_wise=True,
        subject_wise=subject_wise,
    )
    # Average across session
    reliability_voxelwise = np.squeeze(np.nanmean(reliability_voxelwise, axis=0))

    # Set negative reliability values to 0
    reliability_voxelwise[reliability_voxelwise <= 0] = np.nan

    if subject_wise:
        # Make voxels first axis
        reliability_voxelwise = reliability_voxelwise.T
    corr = corr / np.sqrt(reliability_voxelwise)
    return corr


def plot_variability(corr_mean, filename=None, save=True):
    """Plot inter-individual variability and save to file"""
    # Average across subjects
    if corr_mean.ndim > 1:
        corr_mean = np.nanmean(corr_mean, axis=1)

    # Map to surface via nifti
    suit_atlas, _ = am.get_atlas("MNISymC2", ut.base_dir + "/Atlases")
    Nifti = suit_atlas.data_to_nifti(corr_mean)
    surf_data = suit.flatmap.vol_to_surf(Nifti, stats="nanmean", space="MNISymC")

    # Plot
    figsize = (8, 8)
    plt.figure(figsize=figsize)
    suit.flatmap.plot(
        surf_data,
        render="matplotlib",
        new_figure=False,
        cmap="hot",
        cscale=(0, 1),
        overlay_type="func",
        colorbar=True,
        bordersize=4,
    )
    if save:
        plt.savefig(f"{figure_path}/individual_variability/{filename}.png")


def describe_variability():
    norm = True
    subject_wise = True
    sym = "Asym"
    K = 32
    space = "MNISymC2"
    mname = f"Models_03/Nettekoven{sym}{K}_space-{space}"

    # Get individual parcellations
    try:
        probs_indiv = pt.load(f"{ut.model_dir}/Models/{mname}_Uhat.pt")
        probs_info = pd.read_csv(
            f"{ut.model_dir}/Models/{mname}_Uhat_info.tsv", sep="\t"
        )
    except FileNotFoundError:
        probs_indiv, probs_info = sm.export_uhats(mname=mname)
    probs_indiv = probs_indiv.numpy()

    # Get inter-individual variability for all datasets (optional: normalize by reliability)
    datasets = probs_info["dataset"].unique()
    Corr = []
    for dataset in datasets:
        probs_dataset = probs_indiv[probs_info.dataset == dataset, :, :]
        corr_dataset = inter_individual_variability(
            probs_dataset, subject_wise=subject_wise
        )
        Corr.append(corr_dataset)
        if norm:
            corr_dataset = reliability_norm(
                corr_dataset, dataset, subject_wise=subject_wise
            )
            filename = f"{sym}_indiv_var_{dataset}_{K}_norm_subject-{subject_wise}"
        else:
            filename = f"{sym}_indiv_var_{dataset}_{K}_subject-{subject_wise}"

        plot_variability(corr_dataset, filename, save=True)

    # Plot all variabilities as grid
    mean_corr = np.array([np.nanmean(corr, axis=1) for corr in Corr])
    plt.figure(figsize=(14, 8))
    ut.plot_multi_flat(
        mean_corr,
        space,
        grid=(2, 4),
        dtype="func",
        colorbar=True,
        titles=probs_info["dataset"].unique(),
        cmap="hot",
    )
    plt.savefig(
        f"{figure_path}/individual_variability/{sym}_indiv_var_{K}_subject-{subject_wise}.png"
    )

    pass


def plot_dataset_pmaps(plot_parcels=["M1", "M3", "D1", "D2", "D3", "D4"]):
    """Get individual Uhats, average across the Uhats and plot as pmaps"""
    K = 32
    space = "MNISymC2"
    mname = f"Models_03/NettekovenSym{K}_space-{space}"

    # Get individual parcellations
    try:
        probs_indiv, probs_info = pt.load(f"{ut.model_dir}/Models/{mname}_Uhat.pt")
    except FileNotFoundError:
        probs_indiv = sm.export_uhats(mname=mname)
    probs_indiv, probs_info = probs_indiv.numpy()

    fileparts = mname.split("/")
    index, cmap, labels = nt.read_lut(
        ut.base_dir
        + "/..//Cerebellum/ProbabilisticParcellationModel/Atlases/"
        + fileparts[-1].split("_")[0]
        + ".lut"
    )

    labels = labels[1:]
    # Make pmaps for each dataset
    datasets = probs_info["dataset"].unique()
    Corr = []
    for dataset in datasets:
        probs_dataset = probs_indiv[probs_info.dataset == dataset, :, :]
        # Mean prob across subjects for this dataset
        Prob = np.mean(probs_dataset, axis=0)
        subset = [labels.index(p + "L") for p in plot_parcels]
        parc_names = "-".join([str(p) for p in plot_parcels])
        fig = da.save_pmaps(
            Prob,
            labels,
            space,
            subset=subset,
            filename=f"{figure_path}/pmaps_{dataset}_{parc_names}.png",
        )

    pass


if __name__ == "__main__":
    describe_variability()
    # plot_dataset_pmaps(['M1', 'M3', 'D1', 'D2', 'D3', 'D4'])
