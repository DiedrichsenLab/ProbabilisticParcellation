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


plt.figure(figsize=(14, 5))
sym = True
w_cos_symmetric, _, _ = cl.parcel_similarity(model_Sym68, plot=False, sym=sym)
labels, clusters, leaves = cl.agglomative_clustering(
    w_cos_symmetric, sym=sym, method='ward', num_clusters=5)
plt.figure(figsize=(14, 5))
plt.imshow(w_cos_symmetric[leaves, :][:, leaves], cmap='viridis')

similarity, _, _ = cl.parcel_similarity(model_Sym68, plot=False, sym=False)
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.imshow(w_cos_symmetric, cmap='viridis')
plt.title('Wcos Symmetric')
plt.subplot(1, 2, 2)
plt.imshow(similarity, cmap='viridis')
plt.title('Similarity')

W = cm.calc_mds(similarity, center=True)
V = np.array([[-0.3, -0.6, 1], [1, -.6, -.7], [1, 1, 1]]).T
V = cm.make_orthonormal(V)
m = np.array([0.65, 0.65, 0.65])
l = np.array([1, 1, 1])

# cmap = ph.colormap_mds(similarity,target=None,clusters=clusters,gamma=0.3)
# cmap = cm.colormap_mds(W, target=(m, l, V), clusters=clusters, gamma=0.1)
cmap_68_map = ListedColormap(cmap_68)
# labels, clusters, leaves = cl.agglomative_clustering(
#     w_cos_symmetric, sym=sym, cmap=cmap_68_map, labels=labels_68, method='ward', num_clusters=5)

plot = True
K = similarity.shape[0]
sym_sim = (similarity + similarity.T) / 2
dist = squareform(1 - sym_sim.round(5))
Z = linkage(dist, 'ward')

plt.figure()
ax = plt.gca()
R = dendrogram(Z, color_threshold=-1, no_plot=not plot)
leaves = R['leaves']


labels = labels_68[1:68 + 1]
# Remove l and r from labels
labels = [label.strip('L').strip('R') for label in labels]
# Order labels by clustering
zipped_data = zip(labels, leaves)
# Sort the zipped data based on positions
sorted_data = sorted(zipped_data, key=lambda x: x[1])
# Extract the sorted strings
labels = [item[0] for item in sorted_data]

ax.set_xticklabels(labels)
ax.set_ylim((-0.2, 1.5))
cl.draw_cmap(ax, cmap, leaves, sym)


# ----- Custom linkage ------

labels_hem = labels_68[1:int(68 / 2) + 1]
labels_hem = [label.replace('L', '').replace('R', '') for label in labels_hem]
# Invert the order of labels_hem elements
labels_hem = labels_hem[::-1]
cmap_reordered = cmap_68[::-1]

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


R = dendrogram(Z, color_threshold=-1, no_plot=not plot)
leaves = R['leaves']

# Order labels by clustering
labels_leaves = [labels_hem[i] for i in leaves]


plt.figure()
ax = plt.gca()
R = dendrogram(Z, color_threshold=-1, no_plot=not plot)
leaves = R['leaves']
ax.set_xticklabels(labels_leaves)
ax.set_ylim((-0.2, 3))
cl.draw_cmap(ax, ListedColormap(cmap_68), leaves, sym)


# # Specify the desired order for the x-axis
# desired_order = sorted(indices_null)  # Example order

# # Reorder the leaves based on the desired order
# leaves_reordered = [leaves[i] for i in desired_order]
# leaves_reordered = leaves_reordered[1:]

# # Reorder the linkage matrix based on the reordered leaves
# Z_reordered = Z[np.array(leaves_reordered), :]

# # Plot the reordered dendrogram
# dendrogram(Z_reordered)
