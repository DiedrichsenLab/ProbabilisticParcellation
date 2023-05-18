"""
Script to analyze parcels using clustering and colormaps
"""

import pandas as pd
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
from Functional_Fusion.dataset import *
import ProbabilisticParcellation.util as ut
from copy import deepcopy
import ProbabilisticParcellation.learn_fusion_gpu as lf
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as sc
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.functional_profiles as fp
import Functional_Fusion.dataset as ds
import generativeMRF.evaluation as ev

pt.set_default_tensor_type(pt.FloatTensor)


def inspect_model_regions_68():
    mname = f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68'
    info, model = ut.load_batch_best(mname)
    w_cos_sim, cos_sim, _ = cl.parcel_similarity(model, plot=False, sym=False)
    Prob = np.array(model.arrange.marginal_prob())
    vol = Prob.sum(axis=1)
    P = Prob / np.sqrt(np.sum(Prob**2, axis=1).reshape(-1, 1))
    spatial_sim = P @ P.T

    D = pd.read_csv(ut.model_dir + '/Atlases/mixed_assignment_68_16.csv')
    idx = D.parcel_orig_idx
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(w_cos_sim[idx, :][:, idx])
    plt.subplot(1, 2, 2)
    plt.imshow(spatial_sim[idx, :][:, idx])
    P.sum(axis=1)
    vol[idx]
    pass


def inspect_model_regions_32():
    mname = f'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed'
    info, model = ut.load_batch_best(mname)
    w_cos_sim, cos_sim, _ = cl.parcel_similarity(model, plot=False, sym=False)
    Prob = np.array(model.arrange.marginal_prob())
    vol = Prob.sum(axis=1)
    P = Prob / np.sqrt(np.sum(Prob**2, axis=1).reshape(-1, 1))
    spatial_sim = P @ P.T

    D = pd.read_csv(ut.model_dir + '/Atlases/mixed_assignment_68_16.csv')
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(w_cos_sim)
    plt.subplot(1, 2, 2)
    plt.imshow(spatial_sim)
    P.sum(axis=1)
    idx = D.parcel_orig_idx
    vol[idx]
    pass

def individ_parcellation(mname,sn=[0],regions = [29,30],plot='hist'): 
    # Individual training dataset:
    info, model = ut.load_batch_best(mname)
    idata, iinfo, ids = get_dataset(ut.base_dir, 'Mdtb', atlas='MNISymC3',
                                    sess=['ses-s1'], type='CondHalf')

    # Test data set:
    tdata, tinfo, tds = get_dataset(ut.base_dir, 'Mdtb', atlas='MNISymC3',
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

    indx_group = pt.argmax(model.marginal_prob(), dim=0)
    indx_indiv = pt.argmax(Uhat, dim=1)
    indx_data = pt.argmax(emloglik, dim=1)

    uni, countsG = np.unique(indx_group, return_counts=True)
    uni, countsI = np.unique(indx_indiv, return_counts=True)
    countsI = countsI / 24
    D=pd.DataFrame()
    dA=np.zeros((6,len(sn)),dtype=object)
    ind=np.zeros((6,len(sn)),dtype=object)

    for s in sn:
        avrgD = [] 
        avrgD.append(pt.linalg.pinv(m1.emissions[0].X) @ idata[s])
        avrgD.append(pt.linalg.pinv(m1.emissions[1].X) @ tdata[s])
    
        dprime = np.zeros((6,))
    plt.figure(figsize=(17, 6))
    plt.subplot(1, 3, 1)
    calculate_alignment(m1.emissions[0].V, avrgD, indx_data[sn], regions, plot)
    plt.subplot(1, 3, 2)
    calculate_alignment(m1.emissions[0].V, avrgD,
                        indx_indiv[sn], regions, plot)
    plt.subplot(1, 3, 3)
    calculate_alignment(m1.emissions[0].V, avrgD, indx_group, regions, plot)

        for ds in range(2):
            i = ds*3
            dprime[i],dA[i,s],ind[i,s],_ = \
                    calculate_alignment(m1.emissions[ds].V,avrgD[ds],indx_data[s],regions)            
            dprime[1+i],dA[1+i,s],ind[1+i,s],_= \
                    calculate_alignment(m1.emissions[ds].V,avrgD[ds],indx_indiv[s],regions)
            dprime[2+i],dA[2+i,s],ind[2+i,s],ca= \
                    calculate_alignment(m1.emissions[ds].V,avrgD[ds],indx_group,regions)
        d={'SN':[s]*6,
               'region0':[regions[0]]*6,
               'region1':[regions[1]]*6,
               'evaldata':[1,1,1,2,2,2],
               'parcel':['data','indiv','group','data','indiv','group'],
               'dprime':dprime}
        
        D = pd.concat([D,pd.DataFrame(d)])
    
    if plot is not None: 
        plt.figure(figsize=(17,12))    
        for i in range(6):
            dA_plot = np.concatenate(dA[i,:])
            ind_plot = np.concatenate(ind[i,:])
            plt.subplot(2,3,i+1)
            sb.histplot(x=dA_plot,hue=ind_plot,bins=np.linspace(-0.5,0.5,29))
    return D 


def calculate_alignment(V, data, indx, regions, plot='scatter'):
    v = V[:, regions]
    cos_ang = v[:, 0] @ v[:, 1]

    i = (indx == regions[0]) | (indx == regions[1])
    D = data[:, i]
    normD = np.sqrt((D**2).sum(dim=0))
    nD = D/normD
    
    angle = v.T @ nD 
    dA =angle[0]-angle[1] # Difference in angle
    
    m = np.zeros((2,))
    s = np.zeros((2,))

    for r in range(2):
        m[r] = np.nanmean(dA[indx[i]==regions[r]])
        s[r] = np.nanstd(dA[indx[i]==regions[r]])

    dp =(m[0]-m[1])/np.sqrt((s[0]**2+s[1]**2)/2)

    if plot == 'scatter':
        sb.scatterplot(dA, (angle[0] + angle[1]) / 2,
                       hue=indx[i], palette='tab10')
        plt.axvline(cos_ang / 2)
        plt.axvline(-cos_ang / 2)
    elif plot == 'hist':
        sb.histplot(x=dA, hue=indx[i], element='step',
                    palette='tab10',
                    bins=np.linspace(-0.5, 0.5, 29))
        plt.axvline(cos_ang / 2)
        plt.axvline(-cos_ang / 2)
    pass

    return dp,dA,indx[i],cos_ang


if __name__ == "__main__":
    inspect_model_regions_32()
    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-32_meth-mixed'
    # D=individ_parcellation(mname,sn=[21],regions=[28,29],plot='hist')
    D=individ_parcellation(mname,sn=np.arange(24),regions=[28,29],plot='hist')
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
