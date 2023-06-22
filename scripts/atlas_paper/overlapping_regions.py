"""
Script to test 2 overlapping regions to show that these can be separated using individual parcellations
"""

import pandas as pd
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import Functional_Fusion.dataset as ffd
import ProbabilisticParcellation.util as ut
from copy import deepcopy
import ProbabilisticParcellation.plot as ppp
import ProbabilisticParcellation.learn_fusion_gpu as lf
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.functional_profiles as fp
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.evaluation as ev

pt.set_default_tensor_type(pt.FloatTensor)

def inspect_region_overlap(mname='Models_03/NettekovenSym32_space-MNISymC2'):
    """ Check for overlapping regions in 68 atlas
    """
    info, model = ut.load_batch_best(mname)
    w_cos_sim, cos_sim, _ = cl.parcel_similarity(model, plot=False, sym=False)
    Prob = np.array(model.arrange.marginal_prob())
    vol = Prob.sum(axis=1)
    P = Prob / np.sqrt(np.sum(Prob**2, axis=1).reshape(-1, 1))
    spatial_sim = P @ P.T

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(w_cos_sim)
    plt.subplot(1, 2, 2)
    plt.imshow(spatial_sim)
    P.sum(axis=1)

def overlap_analysis(mname='Models_03/NettekovenSym32_space-MNISymC2',
                         subj_n=[0],regions = [29,30],plot='hist'): 
    
    # Individual training dataset:
    info, model = ut.load_batch_best(mname)
    idata, iinfo, ids = ffd.get_dataset(ut.base_dir, 'MDTB', atlas='MNISymC2',
                                    sess=['ses-s1'], type='CondHalf')

    # Test data set:
    tdata, tinfo, tds = ffd.get_dataset(ut.base_dir, 'MDTB', atlas='MNISymC2',
                                    sess=['ses-s2'], type='CondHalf')

    idata = pt.tensor(idata, dtype=pt.float32)
    tdata = pt.tensor(tdata, dtype=pt.float32)

    # Build the individual training model on session 1:
    m1 = deepcopy(model)
    m1.emissions[0].initialize(idata)
    m1.emissions[1].initialize(tdata)
    m1.initialize()

    emloglik = m1.emissions[0].Estep()
    emloglik = emloglik.to(pt.float32)
    Uhat, _ = m1.arrange.Estep(emloglik)

    # Winner take all assignment from group, individual data, and posterior
    indx_group = pt.argmax(model.marginal_prob(), dim=0)
    indx_indiv = pt.argmax(Uhat, dim=1)
    indx_data = pt.argmax(emloglik, dim=1)

    # get counts for each region
    uni, countsG = np.unique(indx_group, return_counts=True)
    uni, countsI = np.unique(indx_indiv, return_counts=True)
    countsI = countsI / len(subj_n)

    # Get the average profiles for the two datsets
    avrgD = np.empty((2,len(subj_n)),dtype=object)
    for s,sn in enumerate(subj_n):
        avrgD[0,s]=(pt.linalg.pinv(m1.emissions[0].X) @ idata[sn]).numpy()
        avrgD[1,s]=(pt.linalg.pinv(m1.emissions[1].X) @ tdata[sn]).numpy()
    
    # Prepare calculation of average alignment 
    D=pd.DataFrame()
    dprime = np.zeros((2,3,len(subj_n)))
    dA=np.zeros((2,3,len(subj_n)),dtype=object)
    cA=np.zeros((2,3,len(subj_n)),dtype=object)
    vA=np.zeros((2,3,len(subj_n)),dtype=object)
    ind=np.zeros((2,3,len(subj_n)),dtype=object)

    for s,sn in enumerate(subj_n):
        for ds in range(2):
            # 0: Purely on the data likelihood
            dprime[ds,0,s],dA[ds,0,s],ind[ds,0,s],cA[ds,0,s],vA[ds,0,s] = \
                    calculate_alignment(m1.emissions[ds].V,avrgD[ds,s],indx_data[sn],regions)
            # 1: Based on individual parcellation
            dprime[ds,1,s],dA[ds,1,s],ind[ds,1,s],cA[ds,1,s],vA[ds,1,s] = \
                    calculate_alignment(m1.emissions[ds].V,avrgD[ds,s],indx_indiv[sn],regions)
            # 2: Based on group parcellation
            dprime[ds,2,s],dA[ds,2,s],ind[ds,2,s],cA[ds,2,s],vA[ds,2,s] = \
                    calculate_alignment(m1.emissions[ds].V,avrgD[ds,s],indx_group,regions)
        d={'SN':[sn]*6,
                'region0':[regions[0]]*6,
                'region1':[regions[1]]*6,
                'evaldata':[1,1,1,2,2,2],
                'parcel':['data','indiv','group','data','indiv','group'],
                'dprime':dprime[:,:,s].flatten()}
    
        D = pd.concat([D,pd.DataFrame(d)])

    if plot is not None: 
        plt.figure(figsize=(17,12))    
        for ds in range(2):
            for t in range(3):
                dA_plot = np.concatenate(dA[ds,t,:])
                ind_plot = np.concatenate(ind[ds,t,:])
                plt.subplot(2,3,ds*3+t+1)
                sb.histplot(x=dA_plot, hue=ind_plot, element='step',
                    palette='tab10',
                    bins=np.linspace(-0.5, 0.5, 29))
                plt.axvline(vA[ds,t,0] / 2)
                plt.axvline(-vA[ds,t,0] / 2)
    return D 


def calculate_alignment(V, data, indx, regions, plot='scatter'):
    """ Calculate the cosine angle between each voxels profile and multiple different V-vectors
    Args:
        V (array): V-matrix
        data (array): data to be compared to V
        indx (array): index of the data 
        regions (array): regions to be compared
        plot (str): 'scatter' or 'hist'
    Returns:
        dprime (float): dprime between the two regions... 
        dA (array): (P,) difference in angle between the two regions
        cosangle (array): (2,P) cos-angle to each region 
        ind (array): region assignment for each voxel
        v_cosang (float): cos-angle between the two regions
    """
    v = V[:, regions].numpy()
    v_cosang = v[:, 0] @ v[:, 1]

    i = (indx == regions[0]) | (indx == regions[1])
    D = data[:, i]
    normD = np.sqrt((D**2).sum(axis=0))
    nD = D/normD
    
    cosang = v.T @ nD 
    dA =cosang[0]-cosang[1] # Difference in angle
    
    m = np.zeros((2,))
    s = np.zeros((2,))

    for r in range(2):
        m[r] = np.nanmean(dA[indx[i]==regions[r]])
        s[r] = np.nanstd(dA[indx[i]==regions[r]])

    dp =(m[0]-m[1])/np.sqrt((s[0]**2+s[1]**2)/2)

    return dp,dA,indx[i],cosang,v_cosang



def plot_alignment(dA,indx,cosang,v_cosang,plot='scatter'):
    if plot == 'scatter':
        sb.scatterplot(dA, (cosang[0] + cosang[1]) / 2,
                       hue=indx, palette='tab10')
        plt.axvline(v_cosang / 2)
        plt.axvline(-v_cosang / 2)
    elif plot == 'hist':
        sb.histplot(x=dA, hue=indx, element='step',
                    palette='tab10',
                    bins=np.linspace(-0.5, 0.5, 29))
        plt.axvline(v_cosang / 2)
        plt.axvline(-v_cosang / 2)
    pass


def make_probmap():
    plt.figure(figsize=(5, 5))
    ppp.plot_parcel_prob('S2','NettekovenSym32',space='MNISymC2',backgroundcolor='w')
    plt.savefig('Prob_S2.png',bbox_inches='tight')
    plt.figure(figsize=(5, 5))
    ppp.plot_parcel_prob('S3','NettekovenSym32',space='MNISymC2',backgroundcolor='w')
    plt.savefig('Prob_S3.png',bbox_inches='tight')



if __name__ == "__main__":
    mname = 'Models_03/NettekovenSym32_space-MNISymC2'
    
    # mname  = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-20'
    
    # make_probmap
    # inspect_region_overlap('Models_03/NettekovenSym32_space-MNISymC2')
    # D=individ_parcellation(mname,sn=[21],regions=[28,29],plot='hist')
    D=overlap_analysis(mname,subj_n=np.arange(24),
                        regions=[28,29],plot='hist')
    plt.figure()
    sb.barplot(data=D,x='evaldata',hue='parcel',y='dprime')
    pass
    # make_NettekovenSym68c32()
    # profile_NettekovenSym68c32()
    # ea.resample_atlas('NettekovenSym68c32',
    #                   atlas='MNISymC2',
    #                   target_space='MNI152NLin6AsymC')
    # Save 3 highest and 2 lowest task maps
    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    # D = query_similarity(mname, 'E3L')
    # save_taskmaps(mname)

    # Merge functionally and spatially clustered scree parcels
    # index, cmap, labels = nt.read_lut(ut.model_dir + '/Atlases/' +
    #                                   fileparts[-1] + '.lut')
    # get data

    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    # f_assignment = 'mixed_assignment_68_16.csv'
    # df_assignment = pd.read_csv(
    #     ut.model_dir + '/Atlases/' + '/' + f_assignment)

    # mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    # f_assignment = 'mixed_assignment_68_16.csv'
    # df_assignment = pd.read_csv(
    #     ut.model_dir + '/Atlases/' + '/' + f_assignment)

    # mapping, labels = mixed_clustering(mname, df_assignment)

    # merge_clusters(ks=[32], space='MNISymC3')
    # export_merged()
    # export_orig_68()

    # --- Export merged models ---
    pass
