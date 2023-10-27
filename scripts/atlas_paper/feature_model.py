import sys
from pathlib import Path
import ProbabilisticParcellation.functional_profiles as fp
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as cm
import Functional_Fusion.dataset as ds
import matplotlib.pyplot as plt
import ProbabilisticParcellation.functional_profiles as fp
import ProbabilisticParcellation.util as ut
from ProbabilisticParcellation.scripts.atlas_paper.ridge_reg import ridgeFit
import nitools as nt
import pandas as pd
import seaborn as sb
import numpy as np
from PcmPy import matrix


def load_profiles(file='NettekovenSym32_profile_individ', regions=None):
    # Load functional profiles
    D = pd.read_csv(ut.export_dir + 'Profiles/' +
                    file + '.tsv', delimiter='\t')
    # Reduce to only MDTB Tasks
    Data = D[D.dataset == 'MDTB']

    if regions is None:
        lut_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/'
        _, cmap, regions = nt.read_lut(lut_dir +
                                       'NettekovenSym32.lut')
        regions = regions[1:]

    Data = Data[['condition'] + regions]
    return Data, regions


def load_features():

    # Load cognitive features
    tags = pd.read_csv(
        f'{ut.model_dir}/Atlases/Profiles/tags/tags_final.tsv', sep="\t"
    )
    # Reduce tags to only MDTB Tasks
    tags = tags[tags.dataset == 'MDTB']
    # Make condition into index
    tags = tags.set_index('condition')

    # Drop everything other than mdtb let hand, right hand and saccade features
    tags_first_tag_column = tags.columns.tolist().index('left_hand_response_execution')
    last_mdtb_tag = tags.columns.tolist().index('saccadic_eye_movement')
    tags = tags.iloc[:, tags_first_tag_column:last_mdtb_tag + 1]
    return tags


def subject_features(tags, Data):
    # Loop through the entries of profile.condition and repeat the tags.condition row
    mdtb_new = [
        "VideoAct",
        "VisualSearchSmall",
        "VisualSearchLarge",
        "SpatialMedDiff",
        "rest",
    ]
    mdtb_old = [
        "VideoActions",
        "VisualSearchEasy",
        "VisualSearchMed",
        "SpatialMapDiff",
        "Rest",
    ]
    mdtb_new2old = dict(zip(mdtb_new, mdtb_old))

    for i, cond in enumerate(Data.condition):
        try:
            row = tags.iloc[tags.index.tolist().index(cond)]
        except:
            row = tags.iloc[tags.index.tolist().index(mdtb_new2old[cond])]
        if i == 0:
            tags_individ = row
        else:
            tags_individ = pd.concat([tags_individ, row], axis=1)
    return tags_individ


def task_indicator(Data):
    # Get index of tasks where each task is a number
    task_codes = {}
    task_indices = []
    for task in Data.condition:
        if task not in task_codes:
            task_codes[task] = len(task_codes) + 1
        task_indices.append(task_codes[task])

    task_matrix = matrix.indicator(task_indices)

    return task_matrix, task_codes


def normalize(Data, tags_task):
    tags_task = (tags_task - np.mean(tags_task, axis=0)) / \
        np.std(tags_task, axis=0)
    Data = (Data - np.mean(Data, axis=0)
            ) / np.std(Data, axis=0)
    return Data, tags_task


def scatter_plot(compare, data, side=None):
    """Compare two regions"""

    if side is not None:
        data = data[data['side'] == side]

    if len(compare[0]) == 1:
        granularity = 'net'
    else:
        granularity = 'reg'

    region1 = data[data[granularity] == compare[0]]
    region2 = data[data[granularity] == compare[1]]
    # Average within each task
    region1 = region1.groupby(['task']).mean().reset_index()
    region2 = region2.groupby(['task']).mean().reset_index()
    fig = plt.scatter(region1['score'], region2['score'])
    # Remove upper and right box lines
    sb.despine()

    # Add labels to the dots
    for i in range(len(region1)):
        plt.annotate(f'{region1.iloc[i].task}',
                     (region1.iloc[i]['score'], region2.iloc[i]['score']))

    # Label
    plt.xlabel(compare[0])
    plt.ylabel(compare[1])
    # Insert lines
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    if side is not None:
        plt.title(side)


def scatter_plot_hemispheres(compare, data):
    """Compare two regions across the hemispheres"""

    region1 = data[data['reg'] == compare][data['side'] == 'L']
    region2 = data[data['reg'] == compare][data['side'] == 'R']
    # Average within each task
    region1 = region1.groupby(['task']).mean().reset_index()
    region2 = region2.groupby(['task']).mean().reset_index()
    fig = plt.scatter(region1['score'], region2['score'])
    # Remove upper and right box lines
    sb.despine()

    # Add labels to the dots
    for i in range(len(region1)):
        plt.annotate(f'{region1.iloc[i].task}',
                     (region1.iloc[i]['score'], region2.iloc[i]['score']))

    # Label
    plt.xlabel(f'Left')
    plt.ylabel(f'Right')
    # Insert lines
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    # plt.title(f'{compare}')


def scatter_plot_tasks(compare, data, ignore_side=False, color='domains'):
    """Compare two tasks across regions
    color: 'domains' or 'hemispheres'
    """

    if ignore_side:
        # average across sides
        data = data[['reg', 'task']].reset_index().groupby(['reg', 'task']).mean().reset_index()
        data['index'] = data['reg'].str[0:2]

    task1 = data[data['task'] == compare[0]]
    task2 = data[data['task'] == compare[1]]
    # Average within each task
    task1 = task1.groupby(['index']).mean().reset_index()
    task2 = task2.groupby(['index']).mean().reset_index()

    # Determine color
    if color == 'domains':
        lut_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/'
        _, cmap, regions = nt.read_lut(lut_dir +
                                       'NettekovenSym32_domain.lut')
        colors = [cmap[regions.index(i[:2])] for i in task1['index']]
    elif color == 'hemispheres':
        cmap = dict(zip(['L', 'R'], ['blue', 'red']))
        colors = [cmap[i[-1]] for i in task1['index']]
    plt.scatter(task1['score'], task2['score'], c=colors)
    # Remove upper and right box lines
    sb.despine()

    # Add labels to the dots
    for i in range(len(task1)):
        plt.annotate(f'{task1.iloc[i]["index"]}',
                     (task1.iloc[i]['score'], task2.iloc[i]['score']))

    # Label
    plt.xlabel(compare[0])
    plt.ylabel(compare[1])
    # Insert lines
    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle='--')
    # plt.title(f'{compare[0]} vs {compare[1]}')


if __name__ == "__main__":
    Data, regions = load_profiles()
    tags = load_features()
    tags_individ = subject_features(tags, Data)
    task_matrix, task_codes = task_indicator(Data)
    tags_task = np.concatenate(
        (tags_individ.T.to_numpy(), task_matrix), axis=1)

    Data_norm, tags_norm = normalize(Data[regions], tags_task)

    # Ridge regression
    R2, features = ridgeFit(Data_norm.to_numpy(), tags_norm,
                            fit_intercept=False, voxel_wise=False, alpha=1.0)
    # Make dataframe
    Features = pd.DataFrame(features.T, columns=[
                            'left_hand', 'right_hand', 'saccades'] + list(task_codes.keys()))

    # Plot
    cmap = plt.get_cmap('RdBu_r')
    plt.figure(figsize=(20, 20))
    plt.imshow(Features, cmap=cmap)
    plt.yticks(np.arange(len(regions)), regions)
    plt.xticks(np.arange(len(Features.columns.tolist())),
               Features.columns.tolist(), rotation=90)

    # Plot a horizontal line in the middle
    plt.hlines(len(regions) / 2 - 0.5, 0,
               len(Features.columns.tolist()), color='black', linewidth=2)
