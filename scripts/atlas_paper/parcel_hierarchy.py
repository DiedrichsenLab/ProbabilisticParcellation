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


def merge_clusters(ks, space="MNISymC3"):
    # save_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/Models/'
    # --- Merge parcels at K=20, 34 & 40 ---
    merged_models = []

    for k in ks:
        mname_fine = f"Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-68"
        mname_coarse = f"Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-{k}"

        # merge model
        _, mname_merged, labels = cl.cluster_parcel(
            mname_fine, mname_coarse, refit_model=True, save_model=True
        )
        merged_models.append(mname_merged)
    return merged_models




def plot_model_taskmaps(
    mname,
    n_highest=3,
    n_lowest=2,
    datasets=["Somatotopic", "Demand"],
    save_task_maps=False,
):
    """Plots taskmaps of highest and lowest scoring tasks for each parcel
    Args:
        n_tasks: Number of tasks to save

    Returns:
        fig: task map plot

    """
    profile = pd.read_csv(
        f'{ut.model_dir}/Atlases/{mname.split("/")[-1]}_task_profile_data.tsv', sep="\t"
    )
    atlas = mname.split("space-")[1].split("_")[0]
    Prob, parcel, _, labels, cmap = ea.analyze_parcel(mname, sym=True)
    labels_sorted = sorted(labels)

    for dataset in datasets:
        # Select dataset
        dataset = datasets[0]
        profile_dataset = profile[profile.dataset == dataset]

        # Get highest scoring task of this dataset
        conditions = profile_dataset.condition
        tasks = {}
        for region in labels_sorted[1:]:
            weights = profile_dataset[region]
            conditions_weighted = [
                (con, w) for con, w in sorted(zip(weights, conditions), reverse=True)
            ]
            high = conditions_weighted[:n_highest]
            low = conditions_weighted[-n_lowest:]
            tasks[region] = high + low
            print(
                f"\n\n\n{region}\nHighest: \t{[el[1] for el in high]}\n\t\t{[el[0] for el in high]}\n\nLowest: \t{[el[1] for el in low]}\n\t\t{[el[0] for el in low]}"
            )

        if save_task_maps:
            # Get task maps
            data, info, _ = ds.get_dataset(ut.base_dir, dataset, atlas=atlas)
            grid = (int(np.ceil((n_highest + n_lowest) / 2)), 2)
            for region in tasks.keys():
                task = tasks[region]
                activity = np.full((len(task), data.shape[2]), np.nan)
                for i, t in enumerate(task):
                    task_name = t[1]
                    activity[i, :] = np.nanmean(
                        data[:, info.cond_name.tolist().index(task_name), :], axis=0
                    )

                titles = [f"{name} V: {np.round(weight,2)}" for weight, name in task]

                cscale = [
                    np.percentile(activity[np.where(~np.isnan(activity))], 5),
                    np.percentile(activity[np.where(~np.isnan(activity))], 95),
                ]
                plot_multi_flat(
                    activity,
                    atlas,
                    grid,
                    cmap="hot",
                    dtype="func",
                    cscale=cscale,
                    titles=titles,
                    colorbar=False,
                    save_fig=True,
                    save_under=f"task_maps/{dataset}_{region}.png",
                )

    pass


def save_taskmaps(mname):
    """Saves taskmaps of highest and lowest scoring tasks for each parcel
    Args:
        n_tasks: Number of tasks to save
        datasets:
    """
    mname = "Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68"

    plt.figure(figsize=(7, 10))
    plot_model_taskmaps(mname, n_highest=3, n_lowest=3)
    plt.savefig(f"tmaps_01.png", format="png")


if __name__ == "__main__":
    # make_NettekovenSym68c32()
    # profile_NettekovenSym68c32()
    # ea.resample_atlas('NettekovenSym68c32',
    #                   atlas='MNISymC2',
    #                   target_space='MNI152NLin6AsymC')
    # Save 3 highest and 2 lowest task maps
    # mname = 'Models_03/NettekovenSym32_space-MNISymC2'
    # D = query_similarity(mname, 'I1L')
    # save_taskmaps(mname)

    # Merge functionally and spatially clustered scree parcels
    # index, cmap, labels = nt.read_lut(ut.model_dir + '/Atlases/' +
    #                                   fileparts[-1] + '.lut')
    # get data

    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    # f_assignment = 'mixed_assignment_68_16.csv'
    # df_assignment = pd.read_csv(
    #     ut.model_dir + '/Atlases/' + '/' + f_assignment)

    mname = "Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68"
    f_assignment = "mixed_assignment_68_16.csv"
    df_assignment = pd.read_csv(ut.model_dir + "/Atlases/" + "/archive/" + f_assignment)

    # mapping, labels = mixed_clustering(mname, df_assignment)

    merge_clusters(ks=[32], space="MNISymC3")
    # export_merged()
    # export_orig_68()

    # --- Export merged models ---

    model_names = [
        "Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_arrange-asym_sep-hem_meth-mixed"
    ]
    # model_names = [
    #     "Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_arrange-asym_sep-hem_reordered_meth-mixed"
    # ]
    # wrong models
    # model_names = [
    #     "Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_arrange-asym_sep-hem_reordered_meth-mixed"
    # ]
    # wrong models
    # model_names = ["Models_03/NettekovenAsym32_space-MNISymC2"]
    info, model = ut.load_batch_best(model_names[0])

    # for model_name in model_names:
    # --- Export merged model ---
    # export_model_merged(model_name)

    # --- Reorder action network ---
    model_names = [
        "Models_03/NettekovenSym68_space-MNISymC2_D5",
        "Models_03/NettekovenAsym68_space-MNISymC2_D5",
        "Models_03/NettekovenSym32_space-MNISymC2_D5",
        "Models_03/NettekovenAsym32_space-MNISymC2_D5",
    ]
    # reorder_action_network(model_names)

    # Export the reordered models (by reordering colour map and labels)
    model_names = [mname.strip("_D5") for mname in model_names]
    for mname in model_names:
        reexport_atlas(mname)

    pass
