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
import ProbabilisticParcellation.scripts.atlas_paper.describe_atlas as da
import ProbabilisticParcellation.evaluate as ppev
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.evaluation as ev
import Functional_Fusion.atlas_map as am
import SUITPy as suit
import os
import nitools as nt
import nibabel as nb

pt.set_default_tensor_type(pt.FloatTensor)


figure_path = (
    "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/papers/AtlasPaper/figure_parts/"
)
if not os.path.exists(figure_path):
    figure_path = "/Users/callithrix/Dropbox/AtlasPaper/figure_parts/"
atlas_dir = (
    "/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/"
)


def plot_variability(corr_mean, filename=None, save=True, cscale=None):
    """Plot inter-individual variability and save to file"""
    # Average across subjects

    # Plot
    figsize = (8, 8)
    plt.figure(figsize=figsize)
    suit.flatmap.plot(
        corr_mean,
        render="matplotlib",
        new_figure=False,
        cmap="hot",
        cscale=cscale,
        overlay_type="func",
        colorbar=True,
        bordersize=4,
    )
    if save:
        plt.savefig(f"{figure_path}/individual_variability/{filename}.png")


def get_rel():
    """Get all task data and reliability maps"""
    T = pd.read_csv(ut.base_dir + "/dataset_description.tsv", sep="\t")
    Data = []
    Reliability = []
    for d, dname in enumerate(T.name[:-1]):
        data, info, dset = ds.get_dataset(ut.base_dir, dname, atlas="MNISymC2")
        Data.append(data)
        # --- Get intra-individual variability (reliability) ---

        reliability_voxelwise, _ = ds.reliability_maps(
            ut.base_dir,
            dname,
            atlas="MNISymC2",
            subtract_mean=True,
            voxel_wise=True,
            subject_wise=True,
        )
        Reliability.append(reliability_voxelwise)

    return Data, Reliability


def get_var(Data, Reliability):
    """Get inter-individual variability for all datasets
    Args:
        Data: list of task data
        Reliability: list of reliability maps
    Returns:
        Corr: list of inter-individual variability maps
        Corr_norm: list of normalized inter-individual variability maps
        Rel: list of reliability maps
    """
    T = pd.read_csv(ut.base_dir + "/dataset_description.tsv", sep="\t")
    Corr, Corr_norm, Rel = [], [], []
    for d, dname in enumerate(T.name[:-1]):
        # Loop over voxels
        corr = []
        corr_norm = []
        reliabilities = []
        for vox in np.arange(Data[0].shape[2]):
            # Correlate across subjects
            corr_vox = np.corrcoef(Data[d][:, :, vox])
            # Get upper triangle
            corr_vox_up = corr_vox[np.triu_indices(corr_vox.shape[0], k=1)]
            # Average upper triangle
            corr_mean = np.nanmean(corr_vox_up)
            # Append to list
            corr.append(corr_mean)

            # --- Get normalized inter-individual variability ---
            # Get reliability for this voxel
            rel = Reliability[d][:, :, vox]
            # If reliability values given per session, average across sessions
            if rel.ndim > 1:
                rel = np.nanmean(rel, axis=0).squeeze()
            # Set negative reliability values to nan
            rel[rel <= 0] = np.nan
            # --- Normalize ---
            # Multiply each element with each other element of rel and take the square root
            rel = np.nanmean(
                np.sqrt(np.outer(rel, rel)[np.triu_indices(rel.shape[0], k=1)])
            )
            corr_normalized = corr_mean / rel
            corr_norm.append(corr_normalized)
            reliabilities.append(np.nanmean(rel))

        # Append to list
        Corr.append(np.array(corr))
        Corr_norm.append(np.array(corr_norm))
        Rel.append(np.array(reliabilities))

    return Corr, Corr_norm, Rel


def calc_variability(Data, Info, subject_wise=False):
    T = pd.read_csv(ut.base_dir + "/dataset_description.tsv", sep="\t")

    # Get inter-individual variability, normalize by geometric mean of reliabilities
    Corr, Corr_norm, Rel = [], [], []
    for d, dname in enumerate(T.name[:-1]):
        # Loop over voxels
        corr = []
        corr_norm = []
        rel = []
        for vox in np.arange(Data[0].shape[2]):
            data = Data[d][:, :, vox]
            correlation = np.corrcoef(data)

            # Get the reliability of each subject by correlating the first and second half
            half1 = data[:, Info[d].half == 1]
            half2 = data[:, Info[d].half == 2]
            reliabilites = []
            for sub in np.arange(half1.shape[0]):
                reliability = np.corrcoef(half1[sub, :], half2[sub, :])[0, 1]
                reliabilites.append(reliability)

            # Set negative reliability values to nan
            reliabilites = np.array(reliabilites)
            reliabilites[reliabilites <= 0] = np.nan
            rel.append(reliabilites)

            # Normalize by geometric mean of reliabilities
            if subject_wise:
                correlation[np.diag_indices(correlation.shape[0])] = np.nan
                noise_ceiling = np.sqrt(np.outer(reliabilites, reliabilites))
                corr_normalised = correlation / noise_ceiling
                # Append to list
                corr.append(np.nanmean(correlation, axis=0))
                corr_norm.append(np.nanmean(corr_normalised, axis=0))

            else:
                correlation = np.nanmean(
                    correlation[np.triu_indices(correlation.shape[0], k=1)]
                )
                noise_ceiling = np.nanmean(
                    np.sqrt(
                        np.outer(reliabilites, reliabilites)[
                            np.triu_indices(reliabilites.shape[0], k=1)
                        ]
                    )
                )
                corr_normalised = correlation / noise_ceiling
                # Append to list
                corr.append(correlation)
                corr_norm.append(corr_normalised)

        # Collect voxels for this dataset
        Corr.append(np.array(corr))
        Corr_norm.append(np.array(corr_norm))
        Rel.append(np.array(rel))

    return Corr, Corr_norm, Rel


def variability_maps():
    T = pd.read_csv(ut.base_dir + "/dataset_description.tsv", sep="\t")
    # Get all task data and reliabilities
    Data, Reliability = get_rel()

    # Get inter-individual variability
    Corr, Corr_norm, Rel = get_var(Data, Reliability)

    # Map to surface via nifti
    suit_atlas, _ = am.get_atlas("MNISymC2", ut.base_dir + "/Atlases")

    # Plot
    for d, dname in enumerate(T.name[:-1]):
        Nifti = suit_atlas.data_to_nifti(Corr_norm[d])
        surf_data = suit.flatmap.vol_to_surf(
            Nifti, stats="nanmean", space="MNISymC")
        plot_variability(
            surf_data,
            filename=f"indiv_var_{dname}_norm",
            save=True,
            cscale=(0, 1),
        )

        Nifti = suit_atlas.data_to_nifti(Corr[d])
        surf_data = suit.flatmap.vol_to_surf(
            Nifti, stats="nanmean", space="MNISymC")
        plot_variability(
            surf_data, filename=f"indiv_var_{dname}", save=True, cscale=(0, 1)
        )

        ifti = suit_atlas.data_to_nifti(Rel[d])
        surf_data = suit.flatmap.vol_to_surf(
            Nifti, stats="nanmean", space="MNISymC")
        plot_variability(
            surf_data, filename=f"indiv_rel_{dname}", save=True, cscale=(0, 1)
        )

    # For stats:
    # Don't get the upper triangle, instead set the diagonal to nan and get the average per row
    # Divide each entry by the square root of reliability of subject 1 times reliability of subject 2
    # --> That's the noise ceiling: geometric mean, the square-root of the product of their reliabilities
    pass


def export_uhats(mname="Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered"):
    """Export Uhats for all subjects in a model"""

    # -- Save individual parcellations --
    prob = ppev.get_individual_parcellation(
        mname, subject="all", dataset=None, session=None
    )

    pt.save(prob, f"{ut.model_dir}/Models/{mname}_Uhat.pt")

    # -- Save info --
    info, model = ut.load_batch_best(mname)
    info = ut.recover_info(info, model, mname)
    participant_info = []
    for dataset in info.datasets:
        dclass = ds.get_dataset_class(ut.base_dir, dataset)
        dataset_participants = dclass.get_participants()
        dataset_participants.loc[:, "dataset"] = dataset
        participant_info.append(
            dataset_participants[["dataset", "participant_id"]])
    participant_info = pd.concat(participant_info)

    participant_info.to_csv(
        f"{ut.model_dir}/Models/{mname}_Uhat_info.tsv", sep="\t", index=False
    )

    # return prob, participant_info  # return Uhats


def plot_dataset_pmaps(plot_parcels=["M1", "M3", "D1", "D2", "D3", "D4"]):
    """Get individual Uhats, average across the Uhats and plot as pmaps"""
    K = 32
    space = "MNISymC2"
    mname = f"Models_03/NettekovenSym{K}_space-{space}"

    # Get individual parcellations
    try:
        probs_indiv, probs_info = pt.load(
            f"{ut.model_dir}/Models/{mname}_Uhat.pt")
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
    # model_variability()
    # plot_dataset_pmaps(['M1', 'M3', 'D1', 'D2', 'D3', 'D4'])
    # # --- Export individual parcellations ---
    # for sym in ["Sym", "Asym"]:
    #     for K in [32, 68]:
    #         mname = f"Models_03/Nettekoven{sym}{K}_space-MNISymC2"
    #         export_uhats(mname)
    # variability_maps()
    T = pd.read_csv(ut.base_dir + "/dataset_description.tsv", sep="\t")
    Data, Info = [], []
    for d, dname in enumerate(T.name[:-1]):
        data, info, dset = ds.get_dataset(ut.base_dir, dname, atlas="MNISymC2")
        Data.append(data)
        Info.append(info)
    Corr, Corr_norm, Rel = calc_variability(Data, Info, subject_wise=True)

    # Export average across all datasets as nifti
    exclude = []
    Corr_norm_mean = np.nanmean([Corr_norm[i] for i in range(
        len(Corr_norm)) if T.name[i] not in exclude], axis=0)

    suit_atlas, _ = am.get_atlas(info.atlas, ut.base_dir + "/Atlases")
    corr_data = suit_atlas.data_to_nifti(Corr_norm_mean)

    save_dir = f"{ut.model_dir}/Models/Evaluation/nettekoven_68/"
    fname = f'individual_variability_group_norm'
    nb.save(corr_data, save_dir + fname + ".nii")
