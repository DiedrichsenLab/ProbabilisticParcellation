""" Analyze functional profiles across emsssion models
"""

import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.matrix as matrix
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as ev
import ProbabilisticParcellation.learn_fusion_gpu as lf
import ProbabilisticParcellation.util as ut
import nitools as nt
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
import torch as pt
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from numpy.linalg import eigh, norm
import matplotlib as mpl
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.patches import Rectangle
from copy import deepcopy
# from wordcloud import WordCloud
import re
import nitools as nt



def get_profile_info(minfo):
    # --- Get conditions ---
    conditions = []
    sessions = []
    datasets = []
    info_l = []
    for d, dname in enumerate(minfo.datasets):
        dataset = ds.get_dataset_class(ut.base_dir, dname)
        T = dataset.get_participants()
        for s in dataset.sessions:
            inf = dataset.get_info(
                s, type=minfo.type[d], subj=T.participant_id, fields=None
            )
            # get condition names: Use only one repetition of each condition
            cs = inf.drop_duplicates(subset=[dataset.cond_ind])
            cs = cs[dataset.cond_name].to_list()
            # remove whitespace and replace underscore with dash
            cs = [c.replace(" ", "").replace("_", "-") for c in cs]

            # append conditions from this session
            dsets = [dname] * len(cs)
            sess = [s] * len(cs)
            conditions.extend(cs)

            sessions.extend(sess)
            datasets.extend(dsets)

    profile_data = pd.DataFrame(
        {"dataset": datasets, "session": sessions, "condition": conditions}
    )
    return profile_data


def get_profiles_model(model, info):
    """Returns the functional profile for each parcel from model V vectors
    Args:
        model: Loaded model
        info: Model info
    Returns:
        parcel_profiles: list of task profiles. Each entry is the V for one emission model
        profile_data: Dataframe of dataset, session and condition info for parcel profiles. Each entry is the dataset name, session name and condition name for the corresponding parcel profile value.
    """
    # --- Get profile ---
    # Get task profile for each emission model and concatenate
    profile = [em.V.numpy() for em in model.emissions]
    p_idx = [em.V.shape[0] for em in model.emissions]
    parcel_profiles = np.concatenate(profile, 0)

    profile_data = get_profile_info(info)
    return parcel_profiles, profile_data


def get_profiles_vermal_lateral(region='A1', source='individ'):
    mname = "Models_03/NettekovenSym32_space-MNISymC2"
    info, model = ut.load_batch_best(mname)
    info = ut.recover_info(info, model, mname)
    # get functional profiles

    # --- Get profile ---
    # Get task profile for each emission model and concatenate
    parcel_profiles = []
    profile_data = []
    data, cond_vec, part_vec, subj_ind, info_ds = lf.build_data_list(
        info.datasets, atlas=info.atlas, type=info.type, join_sess=False
    )
    if source == "individ":
        # Attach the data
        model.initialize(data, subj_ind=subj_ind)
        Uhat, _ = model.Estep()
    elif source == "group":
        Uhat = model.marginal_prob()
        # Repeat for each subject
        T = pd.read_csv(ut.base_dir + "/dataset_description.tsv", sep="\t")
        all_subjects = [
            T[T.name == dset].return_nsubj.values for dset in info.datasets]
        all_subjects = np.cumsum(all_subjects)[-1]
        Uhat = Uhat.repeat(all_subjects, 1, 1)
    Uhat = Uhat.numpy()
    # Split uhat into vermal and lateral component for region
    # Left end of vermis x = 58 (MNISymC2 space x=44)
    # Right end of vermis x = 80 (MNISymC2 space x=33)
    # Import labels
    index, cmap, labels = nt.read_lut(
        ut.model_dir + "/Atlases/NettekovenSym32.lut")
    labels = labels[1:]

    regions_hem = [region + hem for hem in ['L', 'R']]
    index_hem = [labels.index(reg) for reg in regions_hem]

    # Extract region
    Uhat_region = Uhat[:, index_hem, :]
    # Extract vermal and lateral
    suit_atlas, _ = am.get_atlas(info.atlas, ut.base_dir + "/Atlases")
    Uhat_vermal = Uhat_region.copy()
    Uhat_vermal[:, :, (suit_atlas.vox[0, :] < 33) | (
        suit_atlas.vox[0, :] > 44)] = 0

    Uhat_lateral = Uhat_region.copy()
    Uhat_lateral[:, :, (suit_atlas.vox[0, :] > 33) & (
        suit_atlas.vox[0, :] < 44)] = 0

    prof_d = get_profile_info(info)

    Uhat_region = np.concatenate([Uhat_vermal, Uhat_lateral], axis=1)

    parcel_profiles = []
    profile_data = []
    for d, D in enumerate(data):
        # Average data across partitions within session
        C = matrix.indicator(cond_vec[d])
        avrgD = np.linalg.pinv(C) @ D
        # Individual parcellations for this session
        U = Uhat_region[subj_ind[d]]
        T = info_ds[d]["dataset"].get_participants()
        # Get the weighted avarage across the dividual ROIs
        for s in range(subj_ind[d].shape[0]):
            good = ~np.isnan(avrgD[s].sum(axis=0))
            # WEighted sum of the data
            sumD = avrgD[s, :, good].T @ U[s, :, good]
            # Weighted average of the data
            dat = sumD / np.sum(U[s, :, good], axis=0)
            parcel_profiles.append(dat)
            # add data to profile data
            prof_dd = prof_d[
                (prof_d.dataset == info_ds[d]["dname"])
                & (prof_d.session == info_ds[d]["sess"])
            ].copy()
            prof_dd["participant_id"] = [
                T.participant_id.iloc[s]] * prof_dd.shape[0]
            prof_dd["participant_num"] = [0] * prof_dd.shape[0]
            profile_data.append(prof_dd)

    parcel_profiles = np.vstack(parcel_profiles)
    profile_data = pd.concat(profile_data, ignore_index=True)

    # make functional profile dataframe
    labels = [region + '_' + part for part in ['vermal', 'lateral']
              for region in regions_hem]
    parcel_responses = pd.DataFrame(parcel_profiles, columns=labels)
    Prof = pd.concat([profile_data, parcel_responses], axis=1)

    # --- Save profile ---
    # save functional profile as tsv
    mname = mname.split("/")[-1]
    mname = mname.split("_")[0]

    fname = f"{ut.model_dir}/Atlases/Profiles/{mname}_profile_{source}_vermal_lateral.tsv"
    Prof.to_csv(fname, sep="\t", index=False)
    pass


def get_profiles_individ(model, info, dseg=False):
    """Returns the functional profile for each parcel for each subject (unscaled) from data
    Args:
        model: Loaded model
        info: Model info
        desg (bool): Using a hard segmentation? Default False
    Returns:
        parcel_profiles: list of task profiles. Each entry is the V for one emission model
        profile_data: Dataframe of dataset, session and condition info for parcel profiles. Each entry is the dataset name, session name and condition name for the corresponding parcel profile value.
    """
    # --- Get profile ---
    # Get task profile for each emission model and concatenate
    parcel_profiles = []
    profile_data = []
    data, cond_vec, part_vec, subj_ind, info_ds = lf.build_data_list(
        info.datasets, sess=info.sess, atlas=info.atlas, type=info.type, join_sess=False
    )
    # Attach the data
    model.initialize(data, subj_ind=subj_ind)
    Uhat, _ = model.Estep()

    if dseg:
        Uhat = ar.expand_mn(Uhat.argmax(dim=1), Uhat.shape[1])
    Uhat = Uhat.numpy()

    prof_d = get_profile_info(info)

    for d, D in enumerate(data):
        # Average data across partitions within session
        C = matrix.indicator(cond_vec[d])
        avrgD = np.linalg.pinv(C) @ D
        # Individual parcellations for this session
        U = Uhat[subj_ind[d]]
        T = info_ds[d]["dataset"].get_participants()
        # Get the weighted avarage across the dividual ROIs
        for s in range(subj_ind[d].shape[0]):
            good = ~np.isnan(avrgD[s].sum(axis=0))
            # WEighted sum of the data
            sumD = avrgD[s, :, good].T @ U[s, :, good]
            # Weighted average of the data
            dat = sumD / np.sum(U[s, :, good], axis=0)
            parcel_profiles.append(dat)
            # add data to profile data
            prof_dd = prof_d[
                (prof_d.dataset == info_ds[d]["dname"])
                & (prof_d.session == info_ds[d]["sess"])
            ].copy()
            prof_dd["participant_id"] = [
                T.participant_id.iloc[s]] * prof_dd.shape[0]
            prof_dd["participant_num"] = [0] * prof_dd.shape[0]
            profile_data.append(prof_dd)
    return np.vstack(parcel_profiles), pd.concat(profile_data, ignore_index=True)


def get_profiles_group(model, info, dseg=False):
    """Returns the functional profile for each parcel from model V vectors
    Args:
        model: Loaded model
        info: Model info
        desg (bool): Using a hard segmentation? Default False
    Returns:
        parcel_profiles: list of task profiles. Each entry is the V for one emission model
        profile_data: Dataframe of dataset, session and condition info for parcel profiles. Each entry is the dataset name, session name and condition name for the corresponding parcel profile value.
    """
    parcel_profiles = []
    profile_data = []
    data, cond_vec, part_vec, subj_ind, info_ds = lf.build_data_list(
        info.datasets, atlas=info.atlas, type=info.type, join_sess=False
    )
    # Attach the data
    Uhat = model.marginal_prob()

    if dseg:
        Uhat = ar.expand_mn_1d(Uhat.argmax(dim=0), Uhat.shape[0])
    U = Uhat.numpy()

    prof_d = get_profile_info(info)

    for d, D in enumerate(data):
        # Average data across partitions within session
        C = matrix.indicator(cond_vec[d])
        avrgD = np.linalg.pinv(C) @ D
        # Individual parcellations for this session
        T = info_ds[d]["dataset"].get_participants()

        # Get the weighted avarage across the dividual ROIs
        for s in range(subj_ind[d].shape[0]):
            good = ~np.isnan(avrgD[s].sum(axis=0))
            # WEighted sum of the data
            sumD = avrgD[s, :, good].T @ U[:, good].T
            # Weighted average of the data
            dat = sumD / np.sum(U[:, good], axis=1)
            parcel_profiles.append(dat)
            # add data to profile data
            prof_dd = prof_d[
                (prof_d.dataset == info_ds[d]["dname"])
                & (prof_d.session == info_ds[d]["sess"])
            ].copy()
            prof_dd["participant_id"] = [
                T.participant_id.iloc[s]] * prof_dd.shape[0]
            prof_dd["participant_num"] = [0] * prof_dd.shape[0]
            profile_data.append(prof_dd)
    return np.vstack(parcel_profiles), pd.concat(profile_data, ignore_index=True)


def export_profile(
    mname, info=None, model=None, labels=None, source="model", dseg=False, fname=None
):
    """Exports the functional profile for each parcel from model V vectors or data (individ/group)
    Args:
        mname: Model name
        info: Model info
        model: Loaded model
        labels: List of labels for each parcel
        source: Whether to use the 'model','individ','group' to get the profiles
        dseg (bool): Using a hard segmentation? Default False
    """
    if info is None or model is None:
        # Get model
        info, model = ut.load_batch_best(mname)
        info = ut.recover_info(info, model, mname)

    # get functional profiles
    if source == "model":
        parcel_profiles, profile_data = get_profiles_model(
            model=model, info=info)
    elif source == "individ":
        parcel_profiles, profile_data = get_profiles_individ(
            model=model, info=info, dseg=dseg
        )
    elif source == "group":
        parcel_profiles, profile_data = get_profiles_group(
            model=model, info=info, dseg=dseg
        )

    # make functional profile dataframe
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    parcel_responses = pd.DataFrame(parcel_profiles, columns=labels[1:])
    Prof = pd.concat([profile_data, parcel_responses], axis=1)

    # --- Save profile ---
    # save functional profile as tsv
    mname = mname.split("/")[-1]
    mname = mname.split("_")[0]

    if fname is None:
        if dseg == True:
            fname = f"{ut.model_dir}/Atlases/Profiles/{mname}_profile_{source}_dseg.tsv"
        else:
            fname = f"{ut.model_dir}/Atlases/Profiles/{mname}_profile_{source}.tsv"
    Prof.to_csv(fname, sep="\t", index=False)
    return Prof


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


def parcel_profiles(profile, parcels="all", colour_by_dataset=False):
    """Plots wordclouds of functional profiles for each parcel
    Args:
        profile: dataframe with condition information and parcel scores for each condition in each dataset
        parcels: list of parcel labels for which to display the parcel profile or string 'all'. Defaults to 'all'
        colour_by_dataset: Boolean indicating whether condition names should be coloured by originating dataset.

    Returns:
        wc: word cloud object

    """
    if parcels == "all":
        idx_start = list(profile.columns).index("condition") + 1
        idx_end = list(profile.columns).index("dataset_colour") - 1
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
        ax.axis("off")

    return fig


def dataset_colours(
    word, font_size, position, orientation, random_state=None, **kwargs
):
    colour_file = "sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_task_profile_data.tsv"
    profile = pd.read_csv(f"{ut.model_dir}/Atlases/{colour_file}", sep="\t")

    colour = profile[profile.condition == word].dataset_colour.iloc[-1]

    return colour


def cognitive_features(mname):
    """Gets cognitive features for a model"""
    profile = pd.read_csv(
        f'{ut.model_dir}/Atlases/{mname.split("/")[-1]}_profile.tsv', sep="\t"
    )
    first_parcel = profile.columns.tolist().index("condition") + 1
    last_parcel = profile.columns.tolist().index("dataset_colour")
    parcel_columns = profile.columns[first_parcel:last_parcel]
    profile_matrix = profile[parcel_columns].to_numpy()

    feature_dir = f"{ut.model_dir}/Atlases/Profiles/Cognitive_Features"
    features = pd.read_csv(
        f"{ut.model_dir}/Atlases/Profiles/tags/tags.tsv", sep="\t")
    first_feature = features.columns.tolist().index("condition") + 1
    feature_columns = features.columns[first_feature:]
    feature_matrix = features[feature_columns].to_numpy()

    # multiply the profile by the feature matrix
    feature_profile = np.dot(feature_matrix.T, (profile_matrix))

    # make dataframe
    feature_profile = pd.DataFrame(
        feature_profile, columns=parcel_columns, index=feature_columns
    )

    # save dataframe
    feature_profile.to_csv(
        f'{feature_dir}/{mname.split("/")[-1]}_cognitive_features.tsv', sep="\t"
    )

    return feature_profile


def divide_tongue():
    """Divides the tongue region (M2) into vermal and lateral parts, to enable separate examination of the functional profiles"""
    # Divide M2 (tongue region) into vermal and lateral parts
    get_profiles_vermal_lateral(region='M2')
    get_profiles_vermal_lateral(region='M2', source='group')

    source = 'group'
    Prof = pd.read_csv(
        f'{ut.model_dir}/Atlases/Profiles/NettekovenSym32_profile_{source}_vermal_lateral.tsv', sep='\t')
    Prof = Prof.drop(columns=['dataset', 'session',
                     'condition', 'participant_id', 'participant_num'])
    file = f'NettekovenSym32_profile_{source}'
    D = pd.read_csv(ut.export_dir + 'Profiles/' +
                    file + '.tsv', delimiter='\t')
    df = pd.concat([D, Prof], axis=1)
    df.to_csv(f'{ut.model_dir}/Atlases/Profiles/NettekovenSym32_profile_{source}_A1split.tsv',
              sep='\t', index=False)


if __name__ == "__main__":
    short_name = "NettekovenSym32"
    mname = "Models_03/NettekovenSym32_space-MNISymC2"
    info, model = ut.load_batch_best(mname)
    info = ut.recover_info(info, model, mname)
    # # data,inf=get_profiles_individ(model, info)

    fileparts = mname.split("/")
    index, cmap, labels = nt.read_lut(
        ut.model_dir + "/Atlases/" + short_name + ".lut")
    export_profile(mname, info, model, labels, source="group")
    export_profile(mname, info, model, labels, source="group", dseg=True)
    export_profile(mname, info, model, labels, source="individ")
    export_profile(mname, info, model, labels, source="model")

    # features = cognitive_features(mname)
    # profile = pd.read_csv(
    #     f'{ut.model_dir}/Atlases/{mname.split("/")[-1]}_task_profile_data.tsv', sep="\t"
    # )
    # wc = get_wordcloud(profile, selected_region=selected_region)
    # wc.recolor(color_func=dataset_colours)
    # plt.figure()
    # plt.imshow(wc, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()

    # export_profile(mname, info, model, labels, source="individ")

