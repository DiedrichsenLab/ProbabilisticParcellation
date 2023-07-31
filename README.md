# ProbabilisticParcellation
Cerebellar Probabilistic Parcellation project using HierarchBayesParcel and FunctionalFusion

## Dependencies: 
Diedrichsenlab/HierarchicalBayesParcel
Diedrichsenlab/FunctionalFusion
Diedrichsenlab/DCBC

Other dependencies 


## Notebooks / Code to replicate different sections of the paper

### State dependency of functional atlases 

### Fusion outperforms other atlases

### Symmetric and Asymmetric atlasses  

### Fine level of granularity advantageous for individual parcellation


### Hierarchical atlas organisation 

### Characterization of regions

### Cortical Connectivity

### Function and boundary (a)symmetry 


### Individual localization

To get individual parcellations (previously saved as pytorch tensor to save 
time) run the following script):
```scripts.individual_variability.export_uhats(model_name)```

Which Calls: 
```evaluate.get_individual_parcellation(model_name)```

Plotting the individual parcellations: (Fig XA):
```notebooks/individual_parcellation.ipynb```

Calculating and plotting individual variability (Fig XB):
```notebooks/individual_variability.ipynb``` TODO CARO: WHICH ONE ENDED UP IN THE PAPER? HIGHLIGHT IN NOTEBOOK

which calls: 
```scripts.individual_variability.calc_variability(Data, Info, subject_wise=False):```

Plotting probability maps for parcels (Fig XC):
```notebooks/individual_group.ipynb```
```plot.plot_parcel_prob```

Comparing group and individual parcellations with varying length of data (Fig XD):
```notebooks/individual_group.ipynb```




