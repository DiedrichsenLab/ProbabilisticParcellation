""" Analyze functional profiles across emsssion models
"""

import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
from Functional_Fusion.dataset import *
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as ev
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
import ProbabilisticParcellation.util as ut
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh, norm
import matplotlib as mpl
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Rectangle
from copy import deepcopy
from wordcloud import WordCloud
import re


def get_profiles(model, info, norm=False):
    """Returns the functional profile for each parcel
    Args:
        model: Loaded model
        info: Model info
    Returns:
        parcel_profiles: list of task profiles. Each entry is the V for one emission model
        profile_data: Dataframe of dataset, session and condition info for parcel profiles. Each entry is the dataset name, session name and condition name for the corresponding parcel profile value.
    """
    # --- Get profile ---
    # Get task profile for each emission model
    profile = [em.V for em in model.emissions]
    p_idx = [em.V.shape[0] for em in model.emissions]
    parcel_profiles = profile[0]
    for prof in profile[1:]:
        parcel_profiles = pt.cat((parcel_profiles, prof), 0)

    # Normalize the parcel profiles to unit length
    if norm:
        Psq = parcel_profiles**2
        normp = parcel_profiles / \
            np.sqrt(np.sum(Psq.numpy(), axis=1).reshape(-1, 1))

    # --- Get conditions ---
    conditions = []
    sessions = []
    datasets = []
    for d, dname in enumerate(info.datasets):
        _, dinfo, dataset = get_dataset(
            ut.base_dir,
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
                    ut.base_dir,
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


def export_profile(mname, info=None, model=None, labels=None):
    if info is None or model is None:
        # Get model
        info, model = ut.load_batch_best(mname)
        info = ut.recover_info(info, model, mname)

    # get functional profiles
    parcel_profiles, profile_data = get_profiles(model=model, info=info)

    # make functional profile dataframe
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    parcel_responses = pd.DataFrame(
        parcel_profiles.numpy(), columns=labels[1:]
    )
    Prof = pd.concat([profile_data, parcel_responses], axis=1)

    # --- Assign a colour to each dataset (to aid profile visulisation) ---
    datasets = Prof.dataset.unique()
    # get all colours
    all_colours = TABLEAU_COLORS
    rgb = list(all_colours.values())
    colour_names = list(all_colours.keys())

    dataset_colours = dict(zip(datasets, rgb[: len(datasets)]))
    dataset_colour_names = dict(zip(datasets, colour_names[: len(datasets)]))

    Prof['dataset_colour'] = None
    Prof['dataset_colour_name'] = None
    for dataset in datasets:
        Prof.loc[Prof.dataset == dataset,
                 'dataset_colour'] = dataset_colours[dataset]
        Prof.loc[Prof.dataset == dataset,
                 'dataset_colour_name'] = dataset_colours[dataset]

    # --- Save profile ---
    # save functional profile as tsv
    Prof.to_csv(
        f'{ut.model_dir}/Atlases/{mname.split("/")[-1]}_profile.tsv', sep="\t"
    )
    pass


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


def get_wordcloud(profile, selected_region):
    """Plots a wordcloud of condition names where word size is weighted by response vector
    Args:
        profile: dataframe with condition information and parcel scores for each condition in each dataset
        selected_region: region for which to display the parcel profile

    Returns:
        wc: word cloud object

    """
    default_message = {"Select a parcel on the flatmap": 1}
    # When initiliazing the website and if clickin on a null region, show default message
    if selected_region is not None and isinstance(selected_region, str):
        region = selected_region
        weights = profile[region] * 100
        conditions = profile.condition
        conditions_weighted = dict(zip(conditions, weights))
    elif (
        "text" in selected_region["points"][0]
        and selected_region["points"][0]["text"] != "0"
    ):
        # get the region name from clicked parcel
        region = selected_region["points"][0]["text"]
        weights = profile[region] * 100
        conditions = profile.condition
        conditions_weighted = dict(zip(conditions, weights))
    else:
        conditions_weighted = default_message

    # Generate wordcloud image
    wc = WordCloud(background_color="white")
    wc.generate_from_frequencies(conditions_weighted)

    return wc


def parcel_profiles(profile, parcels='all', colour_by_dataset=False):
    """Plots wordclouds of functional profiles for each parcel
    Args:
        profile: dataframe with condition information and parcel scores for each condition in each dataset
        parcels: list of parcel labels for which to display the parcel profile or string 'all'. Defaults to 'all'
        colour_by_dataset: Boolean indicating whether condition names should be coloured by originating dataset.

    Returns:
        wc: word cloud object

    """
    if parcels == 'all':
        idx_start = list(profile.columns).index('condition') + 1
        idx_end = list(profile.columns).index('dataset_colour') - 1
        parcels = sorted(profile.columns[idx_start:idx_end])
    else:
        parcels = sorted(parcels)

    fig = plt.figure(figsize=[40, 80])
    for p, parcel in enumerate(parcels):
        ax = fig.add_subplot(int(np.ceil(len(parcels) / 4)), 4, p + 1)
        wc = get_wordcloud(profile, parcel)
        if colour_by_dataset:
            wc.recolor(color_func=dataset_colours)

        ax.imshow(wc)
        ax.title.set_text(parcel)
        ax.axis('off')

    return fig


def dataset_colours(word, font_size, position, orientation, random_state=None, **kwargs):
    colour_file = "sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_task_profile_data.tsv"
    profile = pd.read_csv(f"{ut.model_dir}/Atlases/{colour_file}", sep="\t")

    colour = profile[profile.condition == word].dataset_colour.iloc[-1]

    return colour


def cognitive_features(mname):
    """Gets cognitive features for a model"""
    profile = pd.read_csv(
        f'{ut.model_dir}/Atlases/{mname.split("/")[-1]}_profile.tsv', sep="\t"
    )
    first_parcel = profile.columns.tolist().index('condition') + 1
    last_parcel = profile.columns.tolist().index('dataset_colour')
    parcel_columns = profile.columns[first_parcel:last_parcel]
    profile_matrix = profile[parcel_columns].to_numpy()

    feature_dir = f'{ut.model_dir}/Atlases/Profiles/Cognitive_Features'
    features = pd.read_csv(
        f'{ut.model_dir}/Atlases/Profiles/tags/tags.tsv', sep="\t")
    first_feature = features.columns.tolist().index('condition') + 1
    feature_columns = features.columns[first_feature:]
    feature_matrix = features[feature_columns].to_numpy()

    # multiply the profile by the feature matrix
    feature_profile = np.dot(feature_matrix.T, (profile_matrix))

    # make dataframe
    feature_profile = pd.DataFrame(
        feature_profile, columns=parcel_columns, index=feature_columns)

    # save dataframe
    feature_profile.to_csv(
        f'{feature_dir}/{mname.split("/")[-1]}_cognitive_features.tsv', sep="\t")

    return feature_profile


if __name__ == "__main__":
    mname = 'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'
    info, model = ut.load_batch_best(mname)
    info = ut.recover_info(info, model, mname)
    fileparts = mname.split('/')
    index, cmap, labels = nt.read_lut(
        ut.model_dir + '/Atlases/' + fileparts[-1] + '.lut')

    export_profile(mname, info, model, labels)
    features = cognitive_features(mname)
    # profile = pd.read_csv(
    #     f'{ut.model_dir}/Atlases/{mname.split("/")[-1]}_task_profile_data.tsv', sep="\t"
    # )
    # wc = get_wordcloud(profile, selected_region=selected_region)
    # wc.recolor(color_func=dataset_colours)
    # plt.figure()
    # plt.imshow(wc, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()
    pass
