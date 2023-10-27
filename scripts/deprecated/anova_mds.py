#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scripts for atlas evaluation
"""
# import ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas as ea
import ProbabilisticParcellation.util as ut
import pandas as pd
from scipy import stats
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

if __name__ == "__main__":

    

    # ---- Load results ----

    with open(f'{ut.model_dir}/Models/Evaluation/nettekoven_68/ARI_granularity.npy', 'rb') as f:
        aris = np.load(f)


    # Normalize aris by within-dataset reliability
    ARI_avg = ea.average_comp_matrix(aris)
    ARI_norm, aris_norm = ea.norm_comp_matrix(aris, ARI_avg)



    # ---- Stats ----
    # Test whether task based datasets are more similar to MDTB than to HCP
    n_parcellations = int(np.sqrt(len(aris)))
    dataset_labels = ['Md', 'Po', 'Ni', 'Ib',
                    'Wm', 'De', 'So', 'Hc']
    mdtb_row = dataset_labels.index('Md')
    hcp_row = dataset_labels.index('Hc')

    mdtb_values = [aris_norm[i * n_parcellations + j]
                for j in np.arange(n_parcellations) for i in np.arange(n_parcellations) if i == mdtb_row and j != mdtb_row and j < hcp_row]
    mdtb_values = [el for arr in mdtb_values for row in arr for el in row]

    hcp_values = [aris_norm[i * n_parcellations + j]
                for j in np.arange(hcp_row) for i in np.arange(n_parcellations) if i == hcp_row and j <= hcp_row and j != mdtb_row  ]
    hcp_values = [el for arr in hcp_values for row in arr for el in row]

    # Print degrees of freedom
    result = stats.ttest_rel(mdtb_values, hcp_values)
    print(result.statistic, result.pvalue)
    df = len(mdtb_values) - 1

    # Print results
    print(f'T = {result.statistic:.3f}, p = {result.pvalue:.3e}, df = {df}')
    # Non scientific notation
    print(f'T = {result.statistic:.3f}, p = {result.pvalue:.3f}, df = {df}')


    all_values = mdtb_values + hcp_values

    # ANOVA with dataset and similarity as factors
    
    # Create a dataframe
    dataframe = pd.DataFrame({'Dataset': list(np.repeat(dataset_labels[1:-1], 25)) + list(np.repeat(dataset_labels[1:-1], 25)),
                            'Similarity': np.repeat(['task', 'rest'], 150),
                            'ari': all_values})
    
    
    # Performing two-way ANOVA
    model = ols('ari ~ C(Dataset) + C(Similarity) +\
    C(Dataset):C(Similarity)',
                data=dataframe).fit()
    result = sm.stats.anova_lm(model, type=2)