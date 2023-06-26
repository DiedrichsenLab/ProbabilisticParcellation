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
import ProbabilisticParcellation.scripts.atlas_paper.describe_atlas as da
import nitools as nt

pt.set_default_tensor_type(pt.FloatTensor)


def reorder_action_network(mnames):
    """Script to reorder the parcels of the action network such that introspection parcels I1 & I2 become action parcels A3 & A4"""

    f_assignment = "mixed_assignment_68_16_4.csv"
    for mname in mnames:
        symmetry = re.findall("[A-Z][^A-Z]*", mname.split("/")[1].split("_")[0])[1][:3]
        # Split on capital letters

        if symmetry.lower() == "sym":
            sym = True
        else:
            sym = False
        if mname.split("/")[1].split("_")[0][-2:] == "32":
            original_idx = "parcel_med_idx_5Domains"
        else:
            original_idx = "parcel_orig_idx_5Domains"

        model_reordered = ea.reorder_model(
            mname,
            sym=sym,
            mname_new=mname.split("_D5")[0],
            assignment=f_assignment,
            original_idx=original_idx,
            save_model=False,
        )


def reexport_atlas(mname):
    """Script to reexport atlas by using the reordered model, importing the exported lut files, reordering the colour map and exporting the atlas that way"""
    f_assignment = "mixed_assignment_68_16_4.csv"

    symmetry = re.findall("[A-Z][^A-Z]*", mname.split("/")[1].split("_")[0])[1][:3]
    # Split on capital letters

    if symmetry.lower() == "sym":
        sym = True
    else:
        sym = False

    # Get reordering index
    df_assignment = pd.read_csv(ut.model_dir + "/Atlases/" + "/" + f_assignment)
    if mname.split("/")[1].split("_")[0][-2:] == "32":
        order = df_assignment["parcel_med_idx_5Domains"].values
        order = order[np.sort(np.unique(order, return_index=True)[1])]
    else:
        order = df_assignment["parcel_orig_idx_5Domains"].values
    order_full = np.concatenate((order, order + len(order)))
    # Import model
    fileparts = mname.split("/")
    info, model = ut.load_batch_best(mname)
    Prob = model.marginal_prob().numpy()

    # Reorder the old labels and cmap files
    index_d5, cmap_d5, labels_d5 = nt.read_lut(
        ut.base_dir
        + "/..//Cerebellum/ProbabilisticParcellationModel/Atlases/"
        + fileparts[-1].split("_")[0]
        + "_D5"
        + ".lut",
    )
    # --- Reorder the cmap file ---
    cmap_new = deepcopy(cmap_d5)
    cmap_new[1:] = cmap_new[1:][order_full]

    # --- Reorder the labels ---
    labels_new = deepcopy(labels_d5)
    labels_new[1:] = np.array(labels_new[1:])[order_full].tolist()

    # Replace I1 & I2 with A3 & A4
    replacements = {
        "I1": "A3",
        "I2": "A4",
    }
    for i, label in enumerate(labels_new):
        if label[:2] in replacements.keys():
            labels_new[i] = replacements[label[:2]] + label[2:]

    # Export the new lut file
    nt.save_lut(
        f'{ut.model_dir}/Atlases/{fileparts[-1].split("_")[0]}.lut',
        np.arange(len(labels_new)),
        cmap_new,
        labels_new,
    )

    # Export the new atlas
    ea.export_map(
        Prob,
        info.atlas,
        ListedColormap(cmap_new),
        labels_new,
        f"{ut.model_dir}/Atlases/{fileparts[-1]}",
    )

    pass


def analyze_parcel(
    mname,
    sym=True,
    num_cluster=5,
    clustering="agglomative",
    cluster_by=None,
    plot=True,
    labels=None,
    weighting=None,
):
    """
    Analyzes the parcellation of a model, and performs clustering on the parcels.

    Args:
    - mname (str): path of the model to be analyzed.
    - sym (bool): whether or not to symmetrize the connectivity matrix. Default is True.
    - num_cluster (int): number of clusters to generate if agglomerative clustering is used. Default is 5.
    - clustering (str): type of clustering to use. Either "agglomative" or "model_guided". Default is "agglomative".
    - cluster_by (str): path to the model to use for guided clustering. Only required if clustering is "model_guided".
    - plot (bool): whether or not to generate plots. Default is True.
    - labels (ndarray): labels for the parcels, if they have already been generated. Default is None.
    - weighting (str): type of weighting to use for calculating parcel similarity. Default is None.

    Returns:
    - Prob (ndarray): the winner-take-all probabilities for each region.
    - parcel (ndarray): the parcel label for each region.
    - atlas (object): the atlas object used for the parcellation.
    - labels (ndarray): the labels for the clusters generated by clustering.
    - cmap (object): the colormap generated for the parcellation.

    """
    # Get model and atlas.
    fileparts = mname.split("/")
    split_mn = fileparts[-1].split("_")
    info, model = ut.load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.arrange.marginal_prob())
    parcel = Prob.argmax(axis=0) + 1

    # Get parcel similarity:
    w_cos_sym, _, _ = cl.parcel_similarity(
        model, plot=True, sym=sym, weighting=weighting
    )

    # Do Clustering:
    if clustering == "agglomative":
        labels_new, clusters, leaves = cl.agglomative_clustering(
            w_cos_sym, sym=sym, num_clusters=num_cluster, plot=False
        )
        while np.unique(clusters).shape[0] < 2:
            num_cluster = num_cluster - 1
            labels_new, clusters, _ = cl.agglomative_clustering(
                w_cos_sym, sym=sym, num_clusters=num_cluster, plot=False
            )
            logging.warning(
                f" Number of desired clusters too small to find at least two clusters. Set number of clusters to {num_cluster} and found {np.unique(clusters).shape[0]} clusters"
            )
    if clustering == "model_guided":
        if cluster_by is None:
            raise ("Need to specify model that guides clustering")
        cluster_info, cluster_model = ut.load_batch_best(cluster_by)
        clusters_half, clusters = cl.guided_clustering(mname, cluster_by)
        labels, cluster_counts = cl.cluster_labels(clusters)
        print(
            f"Found {len(cluster_counts)} clusters with no of regions: {cluster_counts}\n"
        )

    # Make a colormap...
    w_cos_sim, _, _ = cl.parcel_similarity(model, plot=False)
    W = sc.calc_mds(w_cos_sim, center=True)

    # Define color anchors
    m, regions, colors = sc.get_target_points(atlas, parcel)
    cmap = sc.colormap_mds(W, target=(m, regions, colors), clusters=clusters, gamma=0)
    sc.plot_colorspace(cmap(np.arange(model.K)))

    # Replot the Clustering dendrogram, this time with the correct color map
    if clustering == "agglomative":
        if labels is None:
            labels = labels_new
            cl.agglomative_clustering(
                w_cos_sym, sym=sym, num_clusters=num_cluster, plot=True, cmap=cmap
            )

    plt.figure(figsize=(5, 10))
    cl.plot_parcel_size(Prob, cmap, labels, wta=True)

    # Plot the parcellation
    if plot:
        ax = ut.plot_data_flat(
            Prob, atlas.name, cmap=cmap, dtype="prob", labels=labels, render="plotly"
        )
        ax.show()

    return Prob, parcel, atlas, labels, cmap


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


def query_similarity(mname, label):
    labels, w_cos_sim, spatial_sim, ind = cl.similarity_matrices(mname)
    ind = np.nonzero(labels == label)[0]
    D = pd.DataFrame(
        {
            "labels": labels,
            "w_cos_sim": w_cos_sim[ind[0], :],
            "spatial_sim": spatial_sim[ind[0], :],
        }
    )
    return D


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
