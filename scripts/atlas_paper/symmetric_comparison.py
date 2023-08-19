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

def get_individual_sizes(model_pair):
    sym_uhats = pt.load(f"{ut.model_dir}/Models/{model_pair[0]}_Uhat.pt")
    asym_uhats = pt.load(f"{ut.model_dir}/Models/{model_pair[1]}_Uhat.pt")

    # Get winner-take all assignment
    sym_parcels = pt.argmax(sym_uhats, dim=1) + 1
    asym_parcels = pt.argmax(asym_uhats, dim=1) + 1

    # Get parcel sizes
    sym_sumP, sym_sumV = cl.calc_parcel_size(sym_uhats)
    asym_sumP, asym_sumV = cl.calc_parcel_size(asym_uhats)

    return sym_sumP, sym_sumV, asym_sumP, asym_sumV

def plot_counts(sym_counts, asym_counts, labels):
    if sym_counts[0][0].is_integer():
        title = 'Voxel'
    else:
        title = 'Probability Mass'

    # for region_index in np.arange(0, sym_sumV.shape[1]):
    for region_index in np.arange(0, 5):
        # get max for ylim
        # hist, _ = np.histogram(np.concatenate((sym_sumP[:, region_index], asym_sumP[:, region_index])), bins=20)
        hist, _ = np.histogram((sym_sumP[:, region_index]), bins=20)
        ymax = max(hist)


        plt.figure(figsize=(4, 2))  # Adjust figure size as needed
        plt.subplot(1,2,1)
        plt.hist(sym_counts[:, region_index], bins=20, color='blue', alpha=0.7)
        plt.xlabel(f'{title} Count')
        plt.ylabel('N subjects')
        plt.title(f'Symmetric {labels[region_index +1]}')
        plt.ylim([0, ymax+ymax*0.2])
        # Make sure ylim are the same for both plots


        plt.subplot(1,2,2)
        plt.hist(asym_counts[:, region_index], bins=20, color='blue', alpha=0.7)
        plt.xlabel(f'{title} Count')
        plt.ylabel('N subjects')
        plt.title(f'Asymmetric {labels[region_index +1]}')
        plt.ylim([0, ymax+ymax*0.2])

        plt.tight_layout()

        plt.show()


# def plot_counts_hist(sym_counts, asym_counts, labels, subplot=False):
#     if sym_counts[0][0].is_integer():
#         title = 'Voxel'
#     else:
#         title = 'Probability Mass'

#     if subplot:
#         plt.figure(figsize=(10,10))
#         # cols = int(np.ceil(np.sqrt(sym_counts.shape[1])))
#         cols = 4
#         rows = int(np.ceil(sym_counts.shape[1]/cols))
#         for region_index in np.arange(0, sym_sumV.shape[1]):
#             plt.subplot(rows, cols, region_index+1)
#             # for region_index in np.arange(0, 2):
#             sb.histplot(sym_counts[:, region_index], color='blue', alpha=0.5, label='Symmetric')
#             sb.histplot(asym_counts[:, region_index], color='green', alpha=0.5, label='Asymmetric')
#             if (region_index/rows).is_integer():
#                 plt.xlabel(f'{title} Count')
#             if (region_index/cols).is_integer() or region_index==0:
#                 plt.ylabel('N subjects')
            
            
#             # plt.legend()
#             plt.tight_layout()
#             plt.title(f'{labels[region_index +1]}')
#         plt.legend()
#     else:
#         for region_index in np.arange(0, sym_sumV.shape[1]):
#         # for region_index in np.arange(0, 2):
#             sb.histplot(sym_counts[:, region_index], color='blue', alpha=0.5, label='Symmetric')
#             sb.histplot(asym_counts[:, region_index], color='green', alpha=0.5, label='Asymmetric')
#             plt.xlabel(f'{title} Count')
#             plt.ylabel('N subjects')
#             plt.legend()
#             plt.tight_layout()
#             plt.title(f'{labels[region_index +1]}')
#             plt.show()

#     plt.savefig(ut.figure_dir + f"parcel_sizes_sym_vs_asym_indiv.pdf")


def make_df(sym_sumP, sym_sumV, asym_sumP, asym_sumV, labels):
    # subjects = np.stack((np.arange(0, sym_sumV.shape[0]), np.arange(0, sym_sumV.shape[0])))
    # subjects = np.repeat(subjects, 32, axis=0).flatten()
    region_labels = labels[1:]
    regions = np.stack((region_labels, region_labels))
    regions = np.repeat(regions, sym_sumV.shape[0], axis=0).flatten()

    symmetry = np.repeat(np.stack(['sym', 'asym']), sym_sumV.flatten().shape[0], axis=0).flatten()
    subjects = [f'sub-{n+1}' for n in np.repeat(np.arange(0, sym_sumV.shape[0]),len(region_labels))] + [f'sub-{n+1}' for n in np.repeat(np.arange(0, sym_sumV.shape[0]),len(region_labels))]


    df = pd.DataFrame({
    # 'subject': subjects,
    # 'region': list(np.repeat(labels[1:], 222)),
    'subject': subjects,
    'region': regions,
    'cnum': np.repeat(np.stack((np.arange(len(region_labels)) + 1, np.arange(len(region_labels)) + 1)), 111, axis=0).flatten(),
    'voxels': np.concatenate((sym_sumV.flatten(), asym_sumV.flatten())),
    'prob': np.concatenate((sym_sumP.flatten(), asym_sumP.flatten())),
    'symmetry': symmetry
    }
    )
    df['side'] = df['region'].str[-1]
    df['region'] = df['region'].str[:-1]
    df['domain'] = df['region'].str[0]
    return df



def plot_counts_hist(df, plot='prob'):
    # Plot histogram of counts
    # Create a FacetGrid with regions as rows and hemisphere side as columns
    # for d in df.domain.unique():
        # df_d = df[df.domain == d]
        # grid = sns.FacetGrid(df_d, row='region', col='side', hue='symmetry', height=3, aspect=1)
        # figname = f"parcel_sizes_sym_vs_asym_indiv_grid_{plot}_{d}.pdf"
    for r in df.region.unique().str[:-1]
        df_r = df[df.region == r]
        grid = sns.FacetGrid(df_r, row='domain', col='side', hue='symmetry', height=3, aspect=1)
        figname = f"parcel_sizes_sym_vs_asym_indiv_grid_{plot}_{r}.pdf"
        # Map the histograms onto the grid
        grid.map(sns.histplot, plot, bins=10, kde=False)
        # Adding labels and titles
        grid.set_axis_labels('Value', 'Frequency')
        grid.set_titles(row_template="{row_name}", col_template="{col_name}")
        plt.ylim([0, 105])
        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plt.savefig(ut.figure_dir + figname)
    

def group_change():

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


if __name__ == "__main__":


    # Size difference between asymmetric and symmetric atlas version at the individual level
    sym_sumP, sym_sumV, asym_sumP, asym_sumV = get_individual_sizes(model_pair)

    # Get labels and cmap
    fileparts = model_pair[0].split('/')
    index, cmap, labels = nt.read_lut(ut.export_dir + '/' +
                                        fileparts[-1].split('_space')[0] + '.lut')
    
    # Plot histogram of voxel counts
    # plot_counts(sym_sumP, asym_sumP, labels)
    # plot_counts_hist(sym_sumV, asym_sumV, labels, subplot=True)
    df = make_df(sym_sumP, sym_sumV, asym_sumP, asym_sumV, labels)
    plot_counts_hist(df)
    plot_counts_hist(df, plot='voxels')

    voxel_change_indiv = (df[df.symmetry=='asym'].voxels.values - df[df.symmetry=='sym'].voxels.values) / df[df.symmetry=='sym'].voxels.values
    df_change = df[['subject', 'region', 'cnum', 'side', 'domain']][df.symmetry == 'sym']
    df_change['change'] = voxel_change_indiv

    



