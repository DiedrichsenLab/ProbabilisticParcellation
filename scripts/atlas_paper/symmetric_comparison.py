import sys
sys.path.append("..")
import ProbabilisticParcellation.evaluate as ev
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.scripts.atlas_paper.parcel_hierarchy as ph
import ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas as eva
import ProbabilisticParcellation.scripts.atlas_paper.symmetry as sym
import hierarchical_clustering as cl
import Functional_Fusion.dataset as ds
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import torch as pt
import os
import SUITPy as suit
import numpy as np
import Functional_Fusion.atlas_map as am
from matplotlib.colors import ListedColormap
import nitools as nt
import seaborn as sb


figsize = (20, 20)
model_pair = [
        "Models_03/NettekovenSym32_space-MNISymC2",
        "Models_03/NettekovenAsym32_space-MNISymC2",
    ]


atlas = 'MNISymC2'

# Figure settings
figsize = (8, 8)
colorbar = True
bordersize = 4
dtype = 'func'
cmap = 'inferno'
cscale = (0.5, 1)
labels = None

background = 'white'
if background == 'black':
    fontcolor = 'w'
    bordercolor = 'w'
    backgroundcolor = 'k'
elif background == 'white':
    fontcolor = 'k'
    bordercolor = 'k'
    backgroundcolor = 'w'


def get_prob_mass():
    # Get probability mass for each of the left and right parcels of the asymmetric atlas
    # Load asymmetric atlas
    infos, models = [], []
    for mname in model_pair:
        info, model = ut.load_batch_best(mname)
        infos.append(info)
        models.append(model)


    fileparts = mname.split('/')
    index, cmap, labels = nt.read_lut(ut.export_dir + '/' +
                                        fileparts[-1].split('_space')[0] + '.lut')

    prob_sym = models[0].arrange.marginal_prob()
    prob_asym = models[1].arrange.marginal_prob()
    return prob_sym, prob_asym
    

def make_size_subplot(prob_sym, prob_asym, cmap, wta=True):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    D = cl.plot_parcel_size(prob_sym_group, ListedColormap(cmap), labels, wta=wta, sort=False, side='L')
    plt.ylabel('Symmetric regions (L)')

    plt.subplot(2, 2, 2)
    D = cl.plot_parcel_size(prob_sym_group, ListedColormap(cmap), labels, wta=wta, sort=False, side='R')
    plt.ylabel('Symmetric regions (R)')

    plt.subplot(2, 2, 3)
    D = cl.plot_parcel_size(prob_asym_group, ListedColormap(cmap), labels, wta=wta, sort=False, side='L')
    plt.ylabel('Asymmetric regions (L)')

    plt.subplot(2, 2, 4)
    D = cl.plot_parcel_size(prob_asym_group, ListedColormap(cmap), labels, wta=wta, sort=False, side='R')
    plt.ylabel('Asymmetric regions (R)')

    if wta: 
        plt.savefig(ut.figure_dir + f"parcel_sizes_sym_vs_asym_wta.pdf")
    else:
        plt.savefig(ut.figure_dir + f"parcel_sizes_sym_vs_asym.pdf")


def get_change(prob_sym, prob_asym):
    sym_sumP_group, sym_sumV_group = cl.calc_parcel_size(prob_sym)
    asym_sumP_group, asym_sumV_group = cl.calc_parcel_size(prob_asym)
    # Compare probability masses
    mass_change = (asym_sumP_group - sym_sumP_group) / sym_sumP_group
    voxel_change = (asym_sumV_group - sym_sumV_group) / sym_sumV_group
    return mass_change, voxel_change

def plot_change(change, cmap, labels):
    Dvox = pd.DataFrame({'region': labels[1:],
                      'change': change,
                      'cnum': np.arange(change.shape[0]) + 1})

    pal = {d.region: ListedColormap(cmap)(d.cnum) for i, d in Dvox.iterrows()}

    plt.figure(figsize=(10, 10))
    sb.barplot(data=Dvox, y='region', x='change', palette=pal)
    
    if change[0].is_integer():
        plt.savefig(ut.figure_dir + f"parcel_sizes_sym_vs_asym_wta.pdf")
    else:
        plt.savefig(ut.figure_dir + f"parcel_sizes_sym_vs_asym.pdf")



if __name__ == "__main__":

    # Size difference between asymmetric and symmetric atlas version at the group level
    fileparts = model_pair[0].split('/')
    index, cmap, labels = nt.read_lut(ut.export_dir + '/' +
                                        fileparts[-1].split('_space')[0] + '.lut')

    prob_sym_group, prob_asym_group = get_prob_mass()
    make_size_subplot(prob_sym_group, prob_asym_group, cmap, wta=True)
    make_size_subplot(prob_sym_group, prob_asym_group, cmap, wta=False)
    mass_change, voxel_change = get_change(prob_sym_group, prob_asym_group)

    plot_change(mass_change, cmap, labels)
    plot_change(voxel_change, cmap, labels)

    # Size difference between asymmetric and symmetric atlas version at the individual level




