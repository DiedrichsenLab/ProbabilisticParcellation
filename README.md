# ProbabilisticParcellation
Cerebellar Probabilistic Parcellation project using HierarchBayesParcel and FunctionalFusion

## Dependencies: 
Diedrichsenlab/HierarchicalBayesParcel 
Diedrichsenlab/FunctionalFusion
Diedrichsenlab/DCBC
Diedrichsenlab/cortico_cereb_connectivity 

Other dependencies: 

see ```requirements.txt```

## Notebooks / Code to replicate different sections of the paper

### Estimation of the atlas 



### State dependency of functional atlases 



### Fusion outperforms other atlases


Plotting MDS plots for different single dataset parcellations (Fig 1B & 1C & 1D) & Statistics for between-dataset ARI (similarity; normalized to within-dataset ARI) between all datasets and the task-general (MDTB) and rest-based (HCP) data:
```notebooks/evaluate_mds.ipynb```

Plotting DCBC & Statistics DCBC (Fig 1E & 1F):
```notebooks/evaluate_dcbc.ipynb```


### Symmetric and Asymmetric atlasses  
Plotting symmetry (boundary and functional symmetry, i.e. functional lateralization) and comparing asymmetric and symmetric atlas versions:
```notebooks/symmetry.ipynb```



### Fine level of granularity advantageous for individual parcellation



### Hierarchical atlas organisation 



### Characterization of regions



### Cortical Connectivity

Cortical connectivity models are estimated and evaluated in the repository 

```diedrichsenlab/cortico_cereb_connectivity``` denoted ```ccc``` for short.

Models were trained evaluated ```ccc.run_model```, which is called from ```ccc.scripts.scipt_train_eval_models.py```

The final model evaluation results reported in the paper can be found in ```ccc.notebooks.Evaluate_model_int.ipynb```. 

To summarize the connectivity pattern by cerebellar regions: 

```
import cortico_cereb_connectivity.scripts.script_summarize_weights as csw
csw.make_weight_map('Fusion','05')
```

To summarize further by cortical ROI: 

```
T = csw.make_weight_table(dataset="Fusion",extension="06",cortical_roi="yeo17")
```

Summary figures (by yeo17) and full connectivity maps are in the following two notebooks: 
```
cortical_connectivity.ipynb
parcel_summary.ipynb
```

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



### Final export / Production of the atlas

