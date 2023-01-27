""" Analyze functional profiles across emsssion models
"""

import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
from Functional_Fusion.dataset import *
import generativeMRF.emissions as em
import generativeMRF.arrangements as ar
import generativeMRF.full_model as fm
import generativeMRF.evaluation as ev
from scipy.linalg import block_diag
import PcmPy as pcm
import nibabel as nb
import nibabel.processing as ns
import SUITPy as suit
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import sys
import pickle
from ProbabilisticParcellation.util import *
from ProbabilisticParcellation.scripts.parcel_hierarchy import analyze_parcel
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh, norm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from copy import deepcopy
from wordcloud import WordCloud


def recover_info(info, model, mname):
    """Recovers info fields that were lists from tsv-saved strings and adds model type information.
    Args:
        info: Model info loaded form tsv
    Returns:
        info: Model info with list fields.

    """
    # Recover model info from tsv file format
    for var in ["datasets", "sess", "type"]:
        if not isinstance(info[var], list):
            v = eval(info[var])
            if len(model.emissions) > 2 and len(v) == 1:
                v = eval(info[var].replace(" ", ","))
            info[var] = v

    model_settings = {
        "01": [True, True, False],
        "02": [False, True, False],
        "03": [True, False, False],
        "04": [False, False, False],
        "05": [False, True, True],
    }

    info["model_type"] = mname.split("Models_")[1].split("/")[0]
    uniform_kappa = model_settings[info.model_type][0]
    joint_sessions = model_settings[info.model_type][1]

    info["uniform_kappa"] = uniform_kappa
    info["joint_sessions"] = joint_sessions
    return info


def get_profiles(model, info):
    """Returns the functional profile for each parcel
    Args:
        model: Loaded model
        info: Model info
    Returns:
        profile: list of task profiles. Each entry is the V for one emission model
        conditions: list of conditions. Each entry is the condition for one emission model
        conditions_detailed: list of conditions. Each entry is the condition for one emission model along with the dataset name and the session name
    """
    # --- Get profile ---
    # Get task profile for each emission model
    profile = [em.V for em in model.emissions]
    p_idx = [em.V.shape[0] for em in model.emissions]
    parcel_profiles = profile[0]
    for prof in profile[1:]:
        parcel_profiles = pt.cat((parcel_profiles, prof), 0)

    # --- Get conditions ---
    conditions = []
    sessions = []
    datasets = []
    for d, dname in enumerate(info.datasets):
        _, dinfo, dataset = get_dataset(
            base_dir,
            dname,
            atlas=info.atlas,
            sess=info.sess[d],
            type=info.type[d],
            info_only=True,
        )

        if info.joint_sessions:
            cs = dinfo.drop_duplicates(subset=[dataset.cond_ind])
            cs = cs[dataset.cond_name].to_list()
            # remove whitespace and replace underscore with dash
            cs = [c.replace(" ", "").replace("_", "-") for c in cs]

        # if separate emission models for separate sessions, append conditions separated by session
        else:
            cs_sessionwise = []
            for s, ses in enumerate(dataset.sessions):
                # get conditions from selected session
                _, dinfo, dataset = get_dataset(
                    base_dir,
                    dname,
                    atlas=info.atlas,
                    sess=dataset.sessions[s],
                    type=info.type[d],
                    info_only=True,
                )

                cs = dinfo.drop_duplicates(subset=[dataset.cond_ind])
                cs = cs[dataset.cond_name].to_list()
                # remove whitespace and replace underscore with dash
                cs = [c.replace(" ", "").replace("_", "-") for c in cs]

                # append conditions from this session
                sess = [ses] * len(cs)
                dsets = [dname] * len(cs)

                sessions.extend(sess)
                datasets.extend(dsets)
                cs_sessionwise.extend(cs)
                pass

            conditions.extend(cs_sessionwise)

    profile_data = pd.DataFrame(
        {"dataset": datasets, "session": sessions, "condition": conditions}
    )

    return parcel_profiles, profile_data


def plot_wordcloud_dataset(df, dset, region):
    reg = "A1L"
    # When initiliazing the website and if clickin on a null region, show no conditions
    if region is not None and region["points"][0]["text"] != "0":
        # get the region name
        reg = region["points"][0]["text"]
    d = df.conditions[(df.dataset == dset) & (df.label == reg)]
    wc = WordCloud(background_color="white", width=512, height=384).generate(
        " ".join(d)
    )
    return wc.to_image()


def get_wordcloud(parcel_profiles, profile_data, labels, selected_region):
    """Plots a wordcloud of condition names where word size is weighted by response vector
    Args:
        parcel_profiles: parcel scores for each condition in each dataset
        profile_data: condition names of each dataset
        labels: region labels
        selected_region: region for which to display the parcel profile

    Returns:
        wc: word cloud object

    """
    default_message = {"Select a parcel on the flatmap": 1}
    # When initiliazing the website and if clickin on a null region, show default message
    if (
        selected_region is not None
        and "text" in selected_region["points"][0]
        and selected_region["points"][0]["text"] != "0"
    ):
        # get the region name
        region = selected_region["points"][0]["text"]
        conditions = profile_data.condition
        labels = labels.tolist()
        weights = parcel_profiles[:, labels.index(region) - 1] * 100

        conditions_weighted = dict(zip(conditions, weights.numpy()))
    else:
        conditions_weighted = default_message

    # Generate wordcloud image
    wc = WordCloud(background_color="white")
    wc.generate_from_frequencies(conditions_weighted)

    return wc


if __name__ == "__main__":
    # Merge C2 models
    space = "MNISymC2"
    K = 68
    mname = f"Models_03/sym_MdPoNiIbWmDeSo_space-{space}_K-{K}"
    info, model = load_batch_best(mname)
    info = recover_info(info, model, mname)
    # for each parcel, get the highest scoring task
    parcel_profiles, profile_data = get_profiles(model=model, info=info)
    _, _, _, labels, _ = analyze_parcel(mname, sym=True, plot=True)
    wc = get_wordcloud(parcel_profiles, profile_data, labels)

    pass
