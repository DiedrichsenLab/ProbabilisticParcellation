#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripts for atlas evaluation
"""
import ProbabilisticParcellation.evaluate as ev
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.functional_profiles as fp
import Functional_Fusion.dataset as ds
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from scipy import stats
import glob
import numpy as np
import os

# == To generate the eval_all_5existing_on_taskDatasets.tsv file, run the following functions:
# eval_existing in FusionModel/scripts/eval_existing.py

# == To generate the eval_dataset7_sym.tsv and eval_dataset7_asym-hem.tsv files, loop through Ks (10, 20, 34, 40, 68) and datasets run the following functions:
#   -- Evaluate symmetric --
# model_name = ['Md','Po','Ni','Ib','Wm','De','So','MdPoNiIbWmDeSo', 'MdNiIbWmDeSo', 'MdPoIbWmDeSo', 'MdPoNiWmDeSo', 'MdPoNiIbDeSo', 'MdPoNiIbWmSo', 'MdPoNiIbWmDe']
# ks = [10, 20, 34, 40, 68]
# for K in ks:
#   result_5_eval(K=K, symmetric='sym', model_type=['03','04'], model_name=model_name)

#  -- Evaluate asymmetric fitted from symmetric --
# mname_suffix = '_arrange-asym_sep-hem'
# evaluate_models(ks, model_types=['indiv', 'loo', 'all'], model_on=[
# 'task'], test_on='task', mname_suffix=mname_suffix)

def import_existing(cv=False):
    Data = pd.read_csv(
        f'{ut.model_dir}/Models/Evaluation/eval_all_5existing_on_taskDatasets.tsv', sep='\t')
    # Remove non-crossvalidated tests (where test_data == MDTB and train_data == MDTB10)
    if cv:
        for i, row in Data.iterrows():
            if row['test_data'] == 'MDTB' and row['train_data'] == 'MDTB10':
                Data.drop(i, inplace=True)
    Data['train_data_string'] = Data['train_data']
    
    return Data

def import_fusion(cv=False):
    # Import symmetric evaluation
    Sym = pd.read_csv(
        f'{ut.model_dir}/Models/Evaluation/eval_dataset7_sym.tsv', sep='\t')
    Sym['symmetry'] = 'Symmetric'
    Sym['train_data_string'] = Sym['train_data']
    Sym['train_data'] = Sym['train_data'].apply(lambda x: x.split("' '"))

    # Import asymmetric evaluation
    Asym = pd.read_csv(
        f'{ut.model_dir}/Models/Evaluation/eval_dataset7_asym-hem.tsv', sep='\t')
    Asym['symmetry'] = 'Asymmetric'

    # Remove non-crossvalidated tests
    Asym['train_data_string'] = Asym['train_data']
    Asym['train_data'] = Asym['train_data'].apply(lambda x: x.split("', '"))
    # Remove brackets and quotation marks from train_data
    for i, row in Asym.iterrows():
        for j, train_data in enumerate(row['train_data']):
            train_data = train_data.replace('[', '')
            train_data = train_data.replace(']', '')
            train_data = train_data.replace("'", '')
            Asym.at[i, 'train_data'][j] = train_data

    # Remove Asym entries with K=14 and K=28
    Asym = Asym[Asym['K'] != 14]
    Asym = Asym[Asym['K'] != 28]
    
    # Remove non-crossvalidated tests
    if cv:
        for i, row in Asym.iterrows():
            if row['test_data'] in row['train_data'] and not len(row['train_data']) == 7:
                Asym.drop(i, inplace=True)

    
    
    Data = pd.concat([Sym, Asym], axis=0)
    # Data = pd.concat([Data, Sym_add], axis=0)

    # Loop through rows and add indicator for those where train_data list is of length 6
    Data['train_data_len'] = Data['train_data'].apply(lambda x: len(x))
    # Make Training column 'Single' if train_data_len is 1, 'All' if train_data_len is 7 and 'Leave_one_out' if train_data_len is 6
    Data['Training'] = Data['train_data_len'].apply(
        lambda x: 'Single' if x == 1 else 'Leave_one_out')
    Data.loc[Data['train_data_len'] == 7, 'Training'] = 'All'
    Data.loc[Data['train_data_len'] == 6, 'train_data_string'] = 'Leave_one_out'
    Data.loc[Data['train_data_len'] == 7, 'train_data_string'] = 'All'

    # Show How many leave_one_out each K and each symmetry has
    # Drop K=100
    Data[Data['K'] != 100].groupby(
        ['symmetry', 'K', 'Training']).count()['test_data']


    Data[(Data['K'] == 10) & (Data['model_name'] == 'sym_MdPoIbWmDeSo')].groupby(
        ['symmetry', 'atlas', 'model_name', 'Training', 'test_data', 'common_kappa', 'model_type']).count()

    # Remove model_type 04 and K=100
    Data = Data[(Data['model_type'] == 'Models_03') & (Data['K'] != 100)]
    # Show pretty
    # Data[(Data['K'] == 10) & (Data['model_name'] == 'sym_MdPoIbWmDeSo')]

    return Sym, Asym, Data


if __name__ == "__main__":
    
    Existing = import_existing()
    _, _, Fusion = import_fusion()
    # # Get only leave_one_out
    
    # Fusion = Fusion[Fusion['Training'] == 'Leave_one_out']
    
    # Sym = Fusion[Fusion['symmetry'] == 'Symmetric']
    # Asym = Fusion[Fusion['symmetry'] == 'Asymmetric']

    # Remove any entries from Fusion where train_data_string contains a bracket
    Fusion = Fusion[~Fusion['train_data_string'].str.contains("'")]
        
    # Average across subjects
    Existing = Existing[['K', 'train_data_string', 'test_data', 'dcbc_group', 'model_type']].groupby(['K', 'train_data_string', 'test_data', 'model_type']).mean().reset_index()
    Fusion = Fusion[['symmetry', 'K', 'train_data_string', 'test_data', 'dcbc_group', 'model_type']].groupby(['symmetry', 'K', 'train_data_string', 'test_data', 'model_type']).mean().reset_index()

    for symmetry in ['Symmetric', 'Asymmetric']:
        plot_data = Fusion[Fusion['symmetry'] == symmetry]
        plot_data = pd.concat([plot_data, Existing], axis=0)


        plt.figure(figsize=(14, 5))
        sb.barplot(data=plot_data, x='train_data_string', y='dcbc_group')
        plt.savefig(f'{ut.figure_dir}/{symmetry}_vs_ex.png')
        
        print(f'{symmetry} vs MDTB')
        x = plot_data[(plot_data['train_data_string'] == 'All')]['dcbc_group']
        y = plot_data[(plot_data['train_data_string'] == 'MDTB10') ]['dcbc_group']
        result = stats.ttest_ind(x, y)
        print(result)                 
        # print mean of x and y
        print(np.mean(x), np.mean(y))
        
        print(f'{symmetry} vs Buckner17')
        x = plot_data[(plot_data['train_data_string'] == 'All')]['dcbc_group']
        y = plot_data[(plot_data['train_data_string'] == 'Buckner17') ]['dcbc_group']
        result = stats.ttest_ind(x, y)
        print(result)
        print(np.mean(x), np.mean(y))

        # plt.figure()
        # sb.barplot(data=plot_data, x='train_data_string', y='dcbc_group', hue='test_data')
        # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
        # plt.savefig(f'{ut.figure_dir}/{symmetry}_vs_ex_testdata.png')

        for K in [68, 34, 10]:
            plot_data = Fusion[Fusion['symmetry'] == symmetry]
            plot_data = pd.concat([plot_data[plot_data['K'] == K], Existing], axis=0)
            
            plt.figure(figsize=(14, 5))
            sb.barplot(data=plot_data, x='train_data_string', y='dcbc_group')
            plt.savefig(f'{ut.figure_dir}/{symmetry}_vs_ex_K{K}.png')
        
            # plt.figure(figsize=(14, 5))
            # sb.barplot(data=plot_data, x='train_data_string', y='dcbc_group', hue='test_data')
            # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
            # plt.savefig(f'{ut.figure_dir}/{symmetry}_vs_ex_K{K}_testdata.png')

            x = plot_data[(plot_data['train_data_string'] == 'All')]['dcbc_group']
            y = plot_data[(plot_data['train_data_string'] == 'MDTB10')]['dcbc_group']
            print(f'{symmetry} K{K} vs MDTB10')
            result = stats.ttest_ind(x,y)
            print(result)
            print(np.mean(x), np.mean(y))
        
            # Show only MDTB as test_data
            # plt.figure()
            # sb.barplot(data=plot_data[plot_data['test_data'] == 'MDTB'], x='train_data_string', y='dcbc_group')
            # plt.savefig(f'{ut.figure_dir}/{symmetry}_vs_ex_K{K}_testdata_MDTB.png')

            # Test if Leave_one_out is significantly different from Buckner17
            # print(f'{symmetry} K{K} Leave_one_out vs Buckner17')
            # result = stats.ttest_ind(plot_data[(plot_data['train_data_string'] == 'Leave_one_out') & (plot_data['test_data'] == 'MDTB')]['dcbc_group'],
            #                       plot_data[(plot_data['train_data_string'] == 'Buckner17') & (plot_data['test_data'] == 'MDTB')]['dcbc_group'])
            # print(result)

            # print(f'{symmetry} K{K} Leave_one_out vs Buckner17')
            # result = stats.ttest_ind(plot_data[(plot_data['train_data_string'] == 'Leave_one_out') & (plot_data['test_data'] == 'MDTB')]['dcbc_group'],
            #                       plot_data[(plot_data['train_data_string'] == 'Buckner17') & (plot_data['test_data'] == 'MDTB')]['dcbc_group'])
            # print(result)

            # stats.ttest_ind(plot_data[(plot_data['train_data_string'] == 'Leave_one_out') & (plot_data['test_data'] == 'MDTB')]['dcbc_group'],
            #                       plot_data[(plot_data['train_data_string'] == 'Buckner7') & (plot_data['test_data'] == 'MDTB')]['dcbc_group'])

            # stats.ttest_ind(plot_data[(plot_data['train_data_string'] == 'Leave_one_out') & (plot_data['test_data'] == 'MDTB')]['dcbc_group'],
            #                       plot_data[(plot_data['train_data_string'] == 'Ji10') & (plot_data['test_data'] == 'MDTB')]['dcbc_group'])
            

            
        

    pass
