"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
from Functional_Fusion.dataset import *
from scipy.linalg import block_diag
import torch as pt
import matplotlib.pyplot as plt
from ProbabilisticParcellation.util import *
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from copy import deepcopy
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(base_dir).exists():
    base_dir = '/Users/jdiedrichsen/Data/FunctionalFusion/'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))

pt.set_default_tensor_type(pt.FloatTensor)



def analyze_parcel(mname, load_best=True, sym=True):

    fileparts = mname.split('/')
    split_mn = fileparts[-1].split('_')
    if load_best:
        info,model = load_batch_best(mname)
    else:
        info,model = load_batch_fit(mname)

    # Get parcel similarity:
    w_cos_sym,_,_ = cl.parcel_similarity(model,plot=True,sym=sym)

    # groups=['I','L','W','A','O','D','M']
    # Do Clustering:
    num_clusters = 7
    if num_clusters > model.arrange.K:
        num_clusters = model.arrange.K-1
    labels,clusters,leaves = cl.agglomative_clustering(w_cos_sym,sym=sym,num_clusters=4,plot=False)
    ax = plt.gca()

    # Make a colormap
    w_cos_sim,_,_ = cl.parcel_similarity(model,plot=False)
    W = sc.calc_mds(w_cos_sim,center=True)

    # Define color anchors
    m = np.array([0.65,0.65,0.65])

    # Desired orientation of the eigenvectors of MDS in colorspace
    V=np.array([[-0.3,-0.6,1],[1,-.6,-.7],[1,1,1]]).T
    V=sc.make_orthonormal(V)

    cmap = sc.colormap_mds(W,target=(m,V),clusters=clusters,gamma=0)

    # Replot the Clustering dendrogram, this time with the correct color map
    cl.agglomative_clustering(w_cos_sym,sym=sym,num_clusters=7,plot=True,cmap=cmap)
    sc.plot_colorspace(cmap(np.arange(model.K)))

    # Plot the parcellation
    Prob = np.array(model.marginal_prob())
    parcel = Prob.argmax(axis=0)+1
    atlas = split_mn[2][6:]
    ax = plot_data_flat(parcel,atlas,cmap = cmap,
                    dtype='label',
                    labels=labels,
                    render='plotly')
    ax.show()

    return Prob,parcel,atlas,labels,cmap

def make_sfn_atlas():
    Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # Quick hack - hard-code the labels:
    labels = np.array(['0', 'O1L', 'W1L', 'A2L', 'A3L', 'L1L', 'O2L', 'D1L', 'L2L', 'M2L','I1L', 'D2L', 'M3L', 'M4L', 'M1L', 'W4L', 'A1L', 'W2L', 'O1R', 'W1R', 'A2R', 'A3R', 'L1R', 'O2R', 'D1R', 'L2R', 'M2R', 'I1R', 'D2R', 'M3R', 'M4R', 'M1R', 'W4R', 'A1R', 'W2R'], dtype=object)
    ea.export_map(Prob,atlas,cmap,labels,base_dir + '/Atlases/tpl-MNI152NLin2000cSymC/atl-NettekovenSym34')
    ea.resample_atlas('atl-NettekovenSym34')



def merge_clusters():
    save_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/Models/'
    # --- Merge parcels at K=20, 34 & 40 ---
    merged_models = []
    # for k in [10, 14, 20, 28, 34, 40]:
    for k in [14]:

        mname_fine = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'
        mname_coarse = f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-{k}'

        # merge model
        _, mname_merged, mapping = cl.cluster_model(mname_fine, mname_coarse, sym=True, reduce=True)
        merged_models.append(mname_merged)



if __name__ == "__main__":
    
    mname = 'Models_03/sym_De_space-MNISymC2_K-10'
    Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    pass


    # export the merged model
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_merged, load_best=False, sym=True)
    # export_map(Prob,atlas,cmap,labels, save_dir + '/exported/' + mname_merged)

    # # Plot fine, coarse and merged model
    Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_fine,sym=True)
    Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_coarse,sym=True)

    # --- Show Merged Parcellation at K=20, K=34, K=40---
    # mname_fine = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_fine,sym=True)
    # for mname_merged in merged_models:
    #     Prob,parcel,atlas,labels,cmap = analyze_parcel(mname_merged, load_best=False, sym=True)


    # # Show MNISymC2 Parcellation
    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-10'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels,save_dir + '/exported/' + mname)

    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-80'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels, save_dir + '/exported/' + mname)

    # mname = 'Models_04/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-80'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels,save_dir + '/exported/' + mname)

    # --> Model 03, K=68

    # mname = 'Models_03/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-68'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels,save_dir + '/exported/' + mname)

    # mname = 'Models_04/sym_MdPoNiIbWmDeSoHc_space-MNISymC3_K-68'
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # export_map(Prob,atlas,cmap,labels,save_dir + '/exported/' + mname)

    # resample_atlas(mname)
    # make_asymmetry_map(mname)
    # Prob,parcel,atlas,labels,cmap = analyze_parcel(mname,sym=True)
    # cmap = mpl.cm.get_cmap('tab20')
    # rgb=cmap(np.arange(20))
    # plot_colormap(rgb)
    pass

