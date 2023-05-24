# Evaluate cerebellar parcellations
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import HierarchBayesParcel.evaluation as ev
import ProbabilisticParcellation.evaluate as ppev
import matplotlib.pyplot as plt
import seaborn as sb
import sys
from util import *



base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = 'Y:\data\FunctionalFusion'
if not Path(base_dir).exists():
    raise(NameError('Could not find base_dir'))

# Ks = [34]
# for K in Ks:

    
# # Evaluate DCBC
# eval_all_dcbc(model_type=model_type,prefix=sym,K=K,space = 'MNISymC3', models=hcp_models, fname_suffix=fname_suffix)

# Concat DCBC
# concat_all_prederror(model_type=model_type,prefix=sym,K=Ks,outfile=fname_suffix)

evalfile = f'{base_dir}/Models/Evaluation_04/eval_dcbc_asym_K-34_HCPw_asym.tsv'
eval = pd.read_csv(evalfile, sep = '\t')

pass