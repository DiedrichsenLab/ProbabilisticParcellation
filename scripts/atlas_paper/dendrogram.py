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
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform


figure_path = "/Users/jdiedrichsen/Dropbox (Diedrichsenlab)/papers/AtlasPaper/figure_parts/"
if not os.path.exists(figure_path):
    figure_path = "/Users/callithrix/Dropbox/AtlasPaper/figure_parts/"
atlas_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/'

info_Sym68, model_Sym68 = ut.load_batch_best(
    'Models_03/NettekovenSym68_space-MNISymC2')
info_Sym32, model_Sym32 = ut.load_batch_best(
    'Models_03/NettekovenSym32_space-MNISymC2')


# Settings
figsize = (8, 8)
_, cmap_68, labels_68 = nt.read_lut(atlas_dir + 'NettekovenSym68.lut')
_, cmap_32, labels_32 = nt.read_lut(atlas_dir + 'NettekovenSym32.lut')

suit_atlas, _ = am.get_atlas(info_Sym68.atlas, ut.base_dir + '/Atlases')

# ----- Custom linkage ------

labels_hem = labels_68[1:int(68 / 2) + 1]
labels_hem = [label.replace('L', '').replace('R', '') for label in labels_hem]
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

# Construct the pairwise distance matrix
num_points = len(labels_hem)
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


R = dendrogram(Z, color_threshold=-1, no_plot=False)
leaves = R['leaves']

# Order labels by clustering
labels_leaves = [labels_hem[i] for i in leaves]


plt.figure()
ax = plt.gca()
R = dendrogram(Z, color_threshold=-1, no_plot=False)
leaves = R['leaves']
ax.set_xticklabels(labels_leaves)
ax.set_ylim((-0.2, 3))

# Reorder colours apart from first colour (no label colour) for one heimsphere
cmap_reordered = cmap_68[1:(len(cmap_68) - 1) // 2 + 1]
cmap_reordered = cmap_reordered[::-1]
# concatenate numpy arrays in axis 1 cmap_68[0]] + cmap_reordered + cmap_reordered
cmap_reordered = np.concatenate(
    (np.array([cmap_68[0]]), cmap_reordered), axis=0)

cl.draw_cmap(ax, ListedColormap(cmap_reordered), leaves, sym=False)
plt.savefig(figure_path + 'dendrogram_custom.png', dpi=300)
pass
