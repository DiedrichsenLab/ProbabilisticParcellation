import sys
sys.path.append("..")
import ProbabilisticParcellation.evaluate as ev
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.scripts.atlas_paper.parcel_hierarchy as ph
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas as eva
import ProbabilisticParcellation.similarity_colormap as cm
from Functional_Fusion.dataset import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import torch as pt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from scipy.spatial.distance import squareform
import time


figure_path = "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/papers/AtlasPaper/figure_parts/"
if not os.path.exists(figure_path):
    figure_path = "/Users/callithrix/Dropbox/AtlasPaper/figure_parts/"
atlas_dir_export = f'{ut.base_dir}/../Cerebellum/ProbabilisticParcellationModel/Atlases/'

info_Sym68, model_Sym68 = ut.load_batch_best(
    'Models_03/NettekovenSym68_space-MNISymC2')

# Settings
figsize = (8, 8)
_, cmap_68, labels_68 = nt.read_lut(atlas_dir_export + 'NettekovenSym68.lut')
_, cmap_32, labels_32 = nt.read_lut(atlas_dir_export + 'NettekovenSym32.lut')
_, cmap_domain, labels_domain = nt.read_lut(
    atlas_dir_export + 'NettekovenSym68.lut')

suit_atlas, _ = am.get_atlas(info_Sym68.atlas, ut.base_dir + '/Atlases')


def reorder_leaves(Z, leaves_order):
    """
    Reorders the leaves of a hierarchical clustering tree according to the specified order.

    Args:
        Z (ndarray): The linkage matrix representing the hierarchical clustering tree.
                     It has shape (n-1, 4), where n is the number of observations.
                     Each row of Z contains information about a merged cluster,
                     including the indices of the merged clusters, the distance between them,
                     and the size of the new cluster.
        leaves_order (list or ndarray): The desired order of the leaves.
                                        It should contain the indices of the leaves in the original order.

    Returns:
        ndarray: The reordered linkage matrix representing the hierarchical clustering tree.
                 It has shape (n-1, 4), where n is the number of observations.
                 Each row of the returned matrix contains information about a merged cluster,
                 including the indices of the merged clusters, the distance between them,
                 and the size of the new cluster.
    """
    # first, we build a tree
    n = Z.shape[0] + 1
    ch = np.zeros((n - 1, 2))
    for i in range(n - 1):
        ch[i] = (Z[i][0], Z[i][1])  # ch[i] = children of node n+i
    Z = Z.astype('int32')
    new_Z = np.zeros((n - 1, 4), dtype=float)
    p = np.zeros(2 * n - 1, dtype='int32')  # array of parents
    for i in range(n - 1):
        p[Z[i, 0]] = i + n
        p[Z[i, 1]] = i + n

    cnt = 0
    used = [False for i in range(2 * n - 1)]
    top_level = np.array(leaves_order, dtype='int')
    new_top_level = np.array([], dtype='int')
    sizes = np.ones(2 * n - 1)
    # correspondence between new and old inner node numbering
    old_inner_node_number = np.zeros(n - 1, dtype='int')

    height = np.zeros(n - 1, dtype=float)
    for i in range(n - 1):
        height[i] = Z[i][2]  # height[i] corresponds to cluster n+i

    # Measure execution time for building the initial tree
    start_time = time.time()
    while top_level.shape[0] != 1:

        for j in range(top_level.shape[0] - 1):  # top level of builded tree
            node1 = top_level[j]
            node2 = top_level[j + 1]
            node1_old = node1
            node2_old = node2

            # change numbering to old numbers in tree
            if node1_old > n - 1:
                node1_old = old_inner_node_number[node1_old - n]
            if node2_old > n - 1:
                node2_old = old_inner_node_number[node2_old - n]

            if p[node1_old] == p[node2_old]:  # same parent in the old tree
                sizes[n + cnt] = sizes[node1] + sizes[node2]
                new_Z[cnt] = [node1, node2,
                              height[p[node1_old] - n], sizes[n + cnt]]
                new_top_level = np.append(new_top_level, n + cnt)
                used[node1] = True
                used[node2] = True
                old_inner_node_number[cnt] = p[node1_old]
                cnt += 1

                # Print time
                end_time = time.time()
                execution_time = end_time - start_time
                print("Updated z: {:.6f} seconds; {}".format(
                    execution_time, top_level.shape[0]))
                start_time = time.time()
            if not used[node1]:
                new_top_level = np.append(new_top_level, node1)

        if not used[top_level[top_level.shape[0] - 1]]:
            new_top_level = np.append(
                new_top_level, top_level[top_level.shape[0] - 1])

        top_level = new_top_level
        new_top_level = np.array([], dtype='int')

    return new_Z


def get_cluster_indices(labels_hem, reverse=False):

    # ----- Custom linkage ------
    if reverse:
        # Invert the order of labels_hem elements
        labels_hem = labels_hem[::-1]

    labels_domain = [label[0] for label in labels_hem]
    labels_concept = [label[:2] for label in labels_hem]

    # Null level
    indices_null = np.unique(labels_hem, return_inverse=True)[1]
    # Concept level
    string_to_index = {}
    indices_concepts = [string_to_index.setdefault(
        string, len(string_to_index)) for string in labels_concept]
    # Domain level
    string_to_index = {}
    indices_domain = [string_to_index.setdefault(
        string, len(string_to_index)) for string in labels_domain]

    return indices_null, indices_concepts, indices_domain, labels_hem


def linkage_matrix(indices_null, indices_concepts, indices_domain):
    # Construct the pairwise distance matrix
    num_points = len(indices_null)
    dist_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if indices_concepts[i] != indices_concepts[j]:
                dist_matrix[i, j] += 1
            if indices_domain[i] != indices_domain[j]:
                dist_matrix[i, j] += 1
            if indices_null[i] != indices_null[j]:
                dist_matrix[i, j] += 1

    # Convert the distance matrix to condensed form
    condensed_dist_matrix = squareform(dist_matrix)

    # Generate the linkage matrix
    Z = linkage(condensed_dist_matrix)
    R = dendrogram(Z, color_threshold=-1, no_plot=True)

    return Z, R


def get_dendrogram(reverse=True):
    labels_hem_orig = labels_68[1:int(68 / 2) + 1]
    labels_hem_orig = [label.replace('L', '').replace('R', '')
                       for label in labels_hem_orig]
    indices_null, indices_concepts, indices_domain, labels_hem = get_cluster_indices(
        labels_hem_orig, reverse=reverse)

    cmap_leaves = cmap_68[1:(len(cmap_68) - 1) // 2 + 1]
    if reverse:
        # Reorder colours apart from first colour (no label colour) for one heimsphere
        cmap_leaves = cmap_leaves[::-1]

    Z, R = linkage_matrix(indices_null, indices_concepts, indices_domain)

    leaves = R['leaves']
    # Order labels by clustering
    labels_leaves = [labels_hem[i] for i in leaves]

    # Draw the colour panels for the parcels
    cmap_leaves = np.concatenate(
        (np.array([cmap_68[0]]), cmap_leaves), axis=0)

    return Z, R, labels_leaves, cmap_leaves


def plot_dendrogram(Z, labels_leaves, cmap_leaves, save=False, filename='dendogram'):
    # Plot the dendrogram
    plt.figure()
    ax = plt.gca()
    R = dendrogram(Z, color_threshold=-1, no_plot=False)
    leaves = R['leaves']
    ax.set_xticklabels(labels_leaves)
    ax.set_ylim((-0.2, 3))
    # Draw the colour panels for the parcels
    cl.draw_cmap(ax, ListedColormap(cmap_leaves), leaves, sym=False)

    # Save the figure
    if save:
        plt.savefig(figure_path + f'{filename}.pdf', dpi=300)

    # D_leaf_colors = {"attr_1": dflt_col,

    #                  "attr_4": "#B061FF",  # Cluster 1 indigo
    #                  "attr_5": "#B061FF",
    #                  "attr_2": "#B061FF",
    #                  "attr_8": "#B061FF",
    #                  "attr_6": "#B061FF",
    #                  "attr_7": "#B061FF",

    #                  "attr_0": "#61ffff",  # Cluster 2 cyan
    #                  "attr_3": "#61ffff",
    #                  "attr_9": "#61ffff",
    #                  }
    pass


def get_dedogram_custom(save=False, filename='dendrogram_reverse'):
    labels_hem_orig = labels_68[1:int(68 / 2) + 1]
    indices_null, indices_concepts, indices_domain, labels_hem = get_cluster_indices(
        labels_hem_orig, reverse=False)

    cmap_leaves = cmap_68[1:(len(cmap_68) - 1) // 2 + 1]
    Z, R = linkage_matrix(indices_null, indices_concepts, indices_domain)

    leaves = R['leaves']
    # Order labels by clustering
    labels_leaves = [labels_hem[i] for i in leaves]


if __name__ == "__main__":
    Z, R, labels_leaves, cmap_leaves = get_dendrogram(reverse=True)
    # plot_dendrogram(Z, labels_leaves, cmap_leaves,
    #                 save=True, filename='dendrogram_reverse')

    # ----- Reorder the dendogram ------
    labels_hem_orig = labels_68[1:int(68 / 2) + 1]
    labels_hem_orig = [label.replace('L', '').replace('R', '')
                       for label in labels_hem_orig]
    reorder_index = [labels_leaves.index(x) for x in labels_hem_orig]
    # reorder color map
    cmap_reordered = np.array([cmap_leaves[i] for i in reorder_index])

    # Reorder the dendogram
    Z_reordered = reorder_leaves(Z, reorder_index)
    Z_reordered = Z[reorder_index, :]
    pass
