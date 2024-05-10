#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for learning fusion on datasets

Created on 02/15/2023 at 2:16 PM
Author: cnettekoven
"""
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.learn_fusion_gpu as lf
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import Functional_Fusion.matrix as matrix
import nibabel as nb
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.evaluation as ev
import torch as pt
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import time


def fit_asym_from_sym_sep_hem(
    mname="Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68", mname_new=None
):
    """
    Fits an asymmetric model from a symmetric model by loading the model, freezing the emission model,
    and fitting the arrangement model. The resulting asymmetric model and information are saved.

    Args:
        mname (str): The name of the symmetric model to load. Default is 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC3_K-68'.
        mname_new (str): The name of the new asymmetric model to save. If None, a default name is generated.

    Returns:
        tuple: A tuple containing the asymmetric model and the corresponding information.

    """

    # Load model
    inf, m = ut.load_batch_best(mname)
    inf = ut.recover_info(inf, m, mname)

    # Freeze emission model and fit arrangement model
    M, new_info = lf.refit_model(m, inf, fit="arrangement", sym_new="asym")

    # Save new model
    if mname_new is None:
        mname_new = f'{mname.split("/")[0]}asym_{mname.split("sym_")[1]}_arrange-asym'
    with open(f"{ut.model_dir}/Models/{mname_new}.pickle", "wb") as file:
        pickle.dump([M], file)

    # Save new info
    new_info.to_csv(f"{ut.model_dir}/Models/{mname_new}.tsv", sep="\t", index=False)
    print(
        f"Done. Saved asymmetric model as: \n\t{mname_new} \nOutput folder: \n\t{ut.model_dir}/Models/ \n\n"
    )
    return M, new_info


def fit_models(
    ks,
    fit_datasets=["all", "loo", "indiv"],
    rest_included=False,
    verbose=True,
    indiv_on_rest_only=False,
    sym_from_asym=False,
    mname_asym=None,
):
    """
    Fits models with different parameters and datasets.

    Args:
        ks (list): A list of values for the parameter K.
        fit_datasets (list): A list of strings specifying which datasets to fit the models on.
                             Possible values: 'all', 'loo', 'indiv'. Default is ['all', 'loo', 'indiv'].
        rest_included (bool): Flag indicating whether the HCP dataset is included in the fit. Default is False.
        verbose (bool): Flag indicating whether to print verbose output. Default is True.
        indiv_on_rest_only (bool): Flag indicating whether to fit individual datasets on the rest dataset only.
                                   Default is False.
        sym_from_asym (bool): Flag indicating whether to fit symmetric models from asymmetric models. Default is False.
        mname_asym (str): The name of the asymmetric model. Required when sym_from_asym is True.

    """

    ########## Settings ##########
    space = "MNISymC3"  # Set atlas space
    msym = "sym"  # Set model symmetry
    t = "03"  # Set model type

    # -- Build dataset list --
    if rest_included:
        n_dsets = 8  # with HCP
    else:
        n_dsets = 7  # without HCP
    alldatasets = np.arange(n_dsets).tolist()
    loo_datasets = [np.delete(np.arange(n_dsets), d).tolist() for d in alldatasets]
    individual_datasets = [[d] for d in alldatasets]

    dataset_list = []
    if "all" in fit_datasets:
        dataset_list.extend([alldatasets])
    if "loo" in fit_datasets:
        dataset_list.extend(loo_datasets)
    if "indiv" in fit_datasets:
        if indiv_on_rest_only:
            dataset_list.append([7])
        else:
            dataset_list.extend(individual_datasets)

    T = pd.read_csv(ut.base_dir + "/dataset_description.tsv", sep="\t")
    for datasets in dataset_list:
        for k in ks:
            datanames = "".join(T.two_letter_code[datasets])
            wdir = ut.model_dir + f"/Models/Models_{t}"
            fname = f"/{msym}_{datanames}_space-{space}_K-{k}.tsv"

            if not sym_from_asym:
                if not Path(wdir + fname).exists():
                    print(
                        f"fitting model {t} with K={k} in space {space} as {fname}..."
                    )
                    if verbose:
                        ut.report_cuda_memory()
                    lf.fit_all(
                        datasets,
                        k,
                        model_type=t,
                        repeats=100,
                        sym_type=[msym],
                        space=space,
                    )
                else:
                    print(
                        f"model {t} with K={k} in space {space} already fitted as {fname}"
                    )
            else:
                mname = f"Models_{t}/sym_{datanames}_space-{space}_K-{k}"
                mname_asym = f"Models_{t}/asym_{datanames}_space-{space}_K-{k}_arrange-asym_sep-hem"
                if not Path(wdir + f'/{mname_asym.split("/")[1]}.tsv').exists():
                    fit_asym_from_sym_sep_hem(mname=mname, mname_new=mname_asym)
                else:
                    print(
                        f"model {t} with K={k} in space {space} already fitted as {mname_asym}"
                    )


if __name__ == "__main__":
    # ks = [10, 20, 34, 40, 68]
    # ks = [28, 30, 36, 38, 74]
    # ks = [68, 80]
    # ks = [14, 28]
    # fit_models(ks=[32], fit_datasets=['all'], rest_included=False)
    # fit_models(ks=ks, fit_datasets=['indiv', 'loo', 'all'],
    #            rest_included=False, indiv_on_rest_only=False, sym_from_asym=False)
    # fit_models(ks=ks, fit_datasets=['indiv', 'loo', 'all'],
    #            rest_included=False, indiv_on_rest_only=False, sym_from_asym=True)
    # fit_models(ks=ks, fit_datasets=['loo'], rest_included=True)

    # fit_asym_from_sym_sep_hem(
    #     mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68', mname_new='Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem')

    fit_asym_from_sym_sep_hem(
        mname="Models_03/NettekovenSym32_space-MNISymC2",
        mname_new="Models_03/NettekovenAsym32_space-MNISymC2_test",
    )

    # atlas = 'MNISymC3'
    # ks = [10, 20, 34, 40, 68]
    # fit_models(ks=ks, fit_datasets=['indiv', 'loo', 'all'],
    #            rest_included=False, indiv_on_rest_only=False, sym_from_asym=True)

    # ks_additional = [14, 28, 32, 56, 60]
