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
import PcmPy as pcm
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from copy import deepcopy
import ProbabilisticParcellation.learn_fusion_gpu as lf
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


def cluster_model(mname_fine, mname_coarse, sym=True):
    """Merges the parcels of a fine parcellation model according to a coarser model.

    Args:
        mname_fine:     Probabilstic parcellation to merge (fine parcellation)
        mname_caorse:   Probabilstic parcellation that determines how to merge (coarse parcellation)
        sym:            Boolean indicating if model is symmetric. Defaults to True.
        reduce:         Boolean indicating if model should be reduced (empty parcels removed). Defaults to True.

    Returns:
        merged_model:   Merged model. Coarse model containing voxel probabilities of fine model (Clustered fine model)
        mname_merged:   Name of merged model
        mapping:        Mapping of fine parcels to coarse parcels.

    """
    # -- Import models --
    # Import fine model
    fileparts = mname_fine.split('/')
    split_mn = fileparts[-1].split('_')
    finfo, fine_model = load_batch_best(mname_fine)

    # Import coarse model
    fileparts = mname_coarse.split('/')
    split_mn = fileparts[-1].split('_')
    cinfo, coarse_model = load_batch_best(mname_coarse)

    # -- Cluster fine model --
    # Get winner take all assignment for fine model
    fine_probabilities = pt.softmax(fine_model.arrange.logpi, dim=0)

    # Get probabilities of coarse model
    coarse_probabilities = pt.softmax(coarse_model.arrange.logpi, dim=0)

    print(f'--- Assigning {mname_fine.split("/")[1]} to {mname_coarse.split("/")[1]} ---\n ++ Start values ++ \n Fine Model: \t{fine_probabilities.shape[0]} Prob Parcels \n Coarse Model: \t{coarse_probabilities.shape[0]} Prob Parcels')
    
    # Get mapping between fine parcels and coarse parcels
    mapping = cl.guided_clustering(fine_probabilities, coarse_probabilities)
    # Move parcels up
    mapping = np.unique(
        mapping, return_inverse=True)[1]

    # -- Merge model --   
    merged_model = cl.merge_model(fine_model, mapping)

    # Make new info
    new_info = deepcopy(finfo)
    new_info['K_coarse'] = int(cinfo.K)
    new_info['model_type'] = mname_fine.split('/')[0]
    new_info['K_original'] = int(new_info.K)
    new_info['K'] = int((mapping.max() + 1) * 2)

    # Refit reduced model
    new_model, new_info = lf.refit_model(merged_model, new_info)

    # -- Save model --
    mname_merged = f'{mname_fine}_merged-{cinfo.K}'
    # save new model
    with open(f'{model_dir}/Models/{mname_merged}.pickle', 'wb') as file:
        pickle.dump([new_model], file)

    # save new info
    # TODO: Make sure this is a dataframe, not a series
    new_info.to_csv(f'{model_dir}/Models/{mname_merged}.tsv', sep='\t')

    return new_model, mname_merged, mapping

if __name__ == "__main__":
    
    # # Load model to check type
    # mname_fine = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-20'
    # fileparts = mname_fine.split('/')
    # split_mn = fileparts[-1].split('_')
    # finfo,fmodel = load_batch_best(mname_fine)
    # type(fmodel)

    save_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/Models/'
    # --- Merge parcels at K=20, 34 & 40 ---
    merged_models = []
    # for k in [10, 14, 20, 28, 34, 40]:
    for k in [14]:

        mname_fine = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'
        mname_coarse = f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-{k}'


        # f_Prob,f_parcel,f_atlas,f_labels,f_cmap = analyze_parcel(mname_fine,sym=True)
        # c_Prob,c_parcel,c_atlas,c_labels,c_cmap = analyze_parcel(mname_coarse,sym=True)
        
        # merge model
        _, mname_merged, mapping = cluster_model(mname_fine, mname_coarse, sym=True)
        merged_models.append(mname_merged)

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
