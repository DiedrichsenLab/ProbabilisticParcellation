import sys
sys.path.append("..")
from ProbabilisticParcellation.util import *
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.similarity_colormap as cm
import ProbabilisticParcellation.functional_profiles as fp
from Functional_Fusion.dataset import *
import matplotlib.pyplot as plt
import string
from itertools import combinations
import PcmPy as pcm
from copy import deepcopy
import ProbabilisticParcellation.util as ut

base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/srv/diedrichsen/data/FunctionalFusion'
if not Path(base_dir).exists():
    base_dir = '/Users/callithrix/Documents/Projects/Functional_Fusion/'
if not Path(base_dir).exists():
    raise (NameError('Could not find base_dir'))


def inspect_cognitive_tags():
    """Inspecting cognitive tags from IBC dataset and MDTB dataset."""
    # Load profile data

    atlas = 'MNISymC2'
    fine_model = f'/Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68'
    profile = pd.read_csv(
        f'{model_dir}/Atlases/{fine_model.split("/")[-1]}_task_profile_data.tsv', sep="\t"
    )

    # Load IBC feature tags
    # ibc_features = pd.read_csv(
    #     f'{model_dir}/../../ibc/all_contrasts_corr.csv', sep=","
    # )
    ibc_features = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/ibc_cognitive_features.csv')
    # --- Inspect IBC cognitive feature tags ---

    # Find ibc conditions that are used for the fusion atlas
    ibc_profile = profile[profile.dataset == 'IBC'].drop_duplicates(subset=[
                                                                    'condition'])
    ibc = pd.DataFrame(columns=ibc_features.columns)
    ibc_profile['tags'] = None
    column = ibc_profile.columns.tolist().index('tags')
    for c, cond in enumerate(ibc_profile.condition):
        fusion_condition = cond.replace('-', '_')
        match = ibc_features['contrast'].str.match(fusion_condition)
        if len(ibc_features[match == True].contrast) > 2:
            # find exact match
            for ibc_condition in ibc_features[match == True].contrast.tolist():
                if fusion_condition == ibc_condition:
                    row = ibc_features[ibc_features.contrast == ibc_condition]
                    ibc = ibc.append(other=row, ignore_index=True)
                    ibc_profile.at[c, column] = str(row['tags'].tolist()[0])
                    print(ibc_profile.iloc[c]['tags'])
    ibc_profile[['dataset', 'session', 'condition', 'tags']].iloc[30:]

    ibc = ibc.drop_duplicates(subset=['contrast'])

    tags = []
    for t, tag in enumerate(ibc.tags):
        if isinstance(tag, float) or tag == "['']":
            print(f'Condition {ibc.iloc[t].contrast} has no tags')
        else:
            tag = tag.split(']')[0] + ']'
            tags.extend(eval(tag.split("]")[0] + ']'))

    # ibc.to_csv('/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/ibc_features.tsv', sep="\t", index=None)

    # Tag frequency
    unique_tags = set(tags)
    counts = [tags.count(tag) for tag in unique_tags]

    print(
        f'Total number of unique cognitive tags: {len(set(tags))}. Of those, {counts.count(1)} tags are only used once. ')

    # What are the least frequent tags?
    tags_sorted = [tag for _, tag in sorted(
        zip(counts, unique_tags), reverse=True)]
    tag_frequency = sorted(counts, reverse=True)
    for tag, freq in zip(tags_sorted, tag_frequency):
        print(tag, freq)

    print(f'Tags: \n {sorted(set(tags))}')

    # --- Which IBC tags are equivalent to MDTB tags? ---
    mdtb_tags = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/mdtb_featureTable.txt', sep='\t')
    mdtb_tags.iloc[0]
    mdtb = list(mdtb_tags.columns[4:])

    # split tags on capitalized letters
    mdtb_split = []
    for t, tag in enumerate(mdtb):
        mdtb_split.append(re.sub(r"([A-Z])", r" \1", tag).split())

    # find tags that match mdtb parts
    agreed = []
    all = []
    for t, tag in enumerate(mdtb_split):
        tag_matches_all = []
        tag_matches = []

        for tag_part in tag:
            matches = [tag_part.lower() in ibc_tag for ibc_tag in tags_sorted]
            if np.any(matches):
                tag_matches_part = [tags_sorted[m]
                                    for m in np.where(matches)[0].tolist()]
                tag_matches.append(tag_matches_part)
                tag_matches_all.extend(tag_matches_part)
        all.append(sorted(set(tag_matches_all)))

        # look for aggreement between first tag-part matches and second tag-part matches
        agreed_tags = []
        if isinstance(tag, list) and len(tag_matches) > 1:
            for tag_part in tag_matches[0]:
                for tag in tag_part:
                    if tag_part in tag_matches[1]:
                        agreed_tags.append(tag_part)

        agreed.append(set(agreed_tags))

    for m, tag in enumerate(mdtb):
        print(f"{tag}: \t ['{agreed[m]}']\n\n\n")


def get_unique_tags():
    mdtb_tags = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/mdtb_ibc_feature_map.csv')
    ibc_tags = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/ibc_features.tsv', sep="\t")

    tags_ibc = []
    for t, tag in enumerate(ibc_tags.tags):
        if isinstance(tag, float) or tag == "['']":
            print(f'Condition {ibc_tags.iloc[t].contrast} has no tags')
        else:
            ibc_features
            tag = tag.split(']')[0] + ']'
            tags_ibc.extend(eval(tag.split("]")[0] + ']'))

    # get unique ibc tags
    tags_ibc_unique = []
    [tags_ibc_unique.append(tag)
     for tag in tags_ibc if not tag in tags_ibc_unique]

    tags_mdtb = []
    for t, tag in enumerate(mdtb_tags['IBC-Feature Equivalent']):
        if isinstance(tag, float) or tag == "['']":
            print(f'Condition {ibc_features.iloc[t].contrast} has no tags')
        else:
            tag = tag.split(']')[0] + ']'
            tags_mdtb.extend(eval(tag.split("]")[0] + ']'))

    tags_ibc_unique = [
        tag for tag in tags_ibc_unique if not tag in tags_mdtb]

    tags_unique = tags_mdtb + tags_ibc_unique

    cognitive_features = pd.DataFrame(tags_unique)
    cognitive_features.to_csv(f'/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/cognitive_features.tsv',
                              sep="\t")


def get_missing_tags():

    mdtb_tags = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/mdtb_ibc_feature_map.csv')
    ibc_tags = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/ibc_features.tsv', sep="\t")
    # Ensure that all ibc conditions are in the ibc feature tag list
    atlas = 'MNISymC2'
    fine_model = f'/Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68'
    profile = pd.read_csv(
        f'{model_dir}/Atlases/{fine_model.split("/")[-1]}_task_profile_data.tsv', sep="\t"
    )

    tagged_conditions = ibc_tags.contrast.tolist()
    fusion_conditions = profile[profile.dataset == 'IBC'].condition.tolist()

    # find conditions in ibc fusion data that are missing from the ibc feature tags
    missing_rows = [c for c, condition in enumerate(
        fusion_conditions) if not condition.replace('-', '_') in tagged_conditions]
    missing_fusion_conditions = pd.DataFrame(
        profile[profile.dataset == 'IBC'].iloc[missing_rows][["dataset", "session", "condition"]])

    # find conditions that are tagged in fusion data
    tagged_fusion_conditions = [
        condition for condition in fusion_conditions if condition.replace('-', '_') in tagged_conditions]
    # find conditions that are part of the ibc feature tags, but have no tags
    missing_rows = [fusion_conditions.index(condition) for c, condition in enumerate(
        tagged_fusion_conditions) if ibc_tags.iloc[tagged_conditions.index(condition.replace('-', '_'))].tags == "['']"]
    untagged_conditions = pd.DataFrame(
        profile[profile.dataset == 'IBC'].iloc[missing_rows][["dataset", "session", "condition"]])

    missing = missing_fusion_conditions.append(untagged_conditions)
    missing.to_csv(f'/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/missing_ibc_conditions.tsv',
                   sep="\t")

    pass


def get_unique_conditions():
    atlas = 'MNISymC2'
    fine_model = f'/Models_03/sym_MdPoNiIbWmDeSo_space-{atlas}_K-68'
    profile = pd.read_csv(
        f'{model_dir}/Atlases/{fine_model.split("/")[-1]}_task_profile_data.tsv', sep="\t"
    )
    len(profile.condition.unique())

    data = profile.drop_duplicates(subset=['condition'], keep='last')
    data = data[["dataset", "session", "condition"]]
    data.to_csv(f'{model_dir}/Atlases/all_conditions.tsv',
                sep="\t", index=None)

    pass


def compile_tags_selftagged_datasets():
    """Compile tags for Nishimoto, Working Memory, Demand, Somatotopic and IBC."""
    # Load profile data

    tags_all = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/cognitive_features_NiIbWmDeSo.csv', sep=','
    )

    tags_ibc = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/ibc_cognitive_features.csv', sep=','
    )
    ibc_contrasts = tags_ibc.contrast.tolist()

    ibc_translation = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/missing_ibc_conditions_alp.tsv', sep='\t')
    translated_conditions = ibc_translation.condition.tolist()

    tags_filled = deepcopy(tags_all)
    for c, condition in enumerate(tags_filled.Condition):
        if tags_filled.iloc[c].Dataset == 'IBC' and condition in translated_conditions:
            t = translated_conditions.index(condition)
            condition_old = ibc_translation.iloc[t].contrast
            if condition_old in ibc_contrasts:
                i = condition_old.index(condition_old)
                tags_filled.at[c, 'tags'] = tags_ibc.iloc[i].tags
        elif condition.replace('-', '_') in ibc_contrasts:
            i = ibc_contrasts.index(condition.replace('-', '_'))
            tags_filled.at[c, 'tags'] = tags_ibc.iloc[i].tags
        else:
            print(f'not found: \t{condition}')

    tags_filled.to_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/cognitive_features_NiIbWmDeSo_filled_1.csv', sep='\t', index=None)

    return tags_filled


def compile_tags_all_datasets():
    """Import cognitive features for all datasets."""
    # --- Load MDTB Tags ---
    # Load data
    tags_mdtb = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/cognitive_features_mdtb.txt', sep='\t'
    )
    # remove superfluous columns
    tags_mdtb = tags_mdtb.drop(columns=['taskNumUni', 'condNumUni'])

    # remove superfluous brackets in column names
    mdtb_columns = [eval(column)[0] if '[' in column else column
                    for column in tags_mdtb.columns.tolist()]
    tags_mdtb.columns = mdtb_columns

    # find all columns in mdtb dataframe that contain a tag
    first_tag = 'left_hand_response_execution'
    t_idx = mdtb_columns.index(first_tag)

    # --- Load All Other Tags ---
    pontine_mdtb_taskmap = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/mdtb_pontine_task_matching.txt', sep='\t'
    )

    profile = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-32_meth-mixed_task_profile_data.tsv', sep='\t')

    # Collect tags of Nishimoto, Working Memory, Demand, Somatotopic and IBC
    tags_other = pd.read_csv(
        '/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/cognitive_features_NiIbWmDeSo.csv', sep=','
    )

    for t, tags in enumerate(tags_other.tags):
        tags_other.tags[t] = eval(tags)

    # --- Get Unique Tags ---
    # Make a list of all unique tags, starting with the mdtb tags
    all_tags = mdtb_columns[t_idx:]
    for tags in tags_other.tags:
        for tag in tags:
            if tag not in all_tags:
                all_tags.append(tag)

    # --- Make Tags into indicators ---
    # Make tags into indicators
    tags_other['tags_other'] = tags_other.tags.apply(
        lambda x: [1 if tag in x else 0 for tag in all_tags])
    tags_other = tags_other.drop(columns=['tags'])

    # --- Concatenate Pontine and MDTB Tags ---
    tags = deepcopy(profile[['dataset', 'session', 'condition']])
    tags = tags.reindex(columns=tags.columns.tolist() +
                        all_tags)

    mdtb_conditions = tags_mdtb.conditionName.tolist()

    # ignore conditions that should not be included
    ignore_conditions = ['p-startup', 'neutral']

    # --- Map old MDTB condition names to new MDTB condition names ---
    # gotta take care of renaming mdtb conditions (new names:  VideoAct, VisualSearchSmall, VisualSearchLarge, SpatialMedDiff)
    mdtb_new = ['VideoAct', 'VisualSearchSmall',
                'VisualSearchLarge', 'SpatialMedDiff', 'rest']
    mdtb_old = ['VideoActions', 'VisualSearchEasy',
                'VisualSearchMed', 'SpatialMapDiff', 'Rest']
    mdtb_new2old = dict(zip(mdtb_new, mdtb_old))

    # --- Initialize empty tags for mdtb conditions ---
    tags_mdtb_zeros = pd.Series(
        [0] * (len(all_tags) - len(mdtb_columns[t_idx:])), index=all_tags[len(mdtb_columns[t_idx:]):])

    # --- Compile tags for all datasets ---
    for r, row in tags.iterrows():
        # extract dataset, session and condition
        dset, ses, cond = row[['dataset', 'session', 'condition']].values
        tag = []
        if dset == 'Pontine' and not cond == 'flexion-extension':
            eq = pontine_mdtb_taskmap[pontine_mdtb_taskmap.condition ==
                                      cond].mdtb_equivalent_task.item()

            if eq not in mdtb_conditions and eq + 'Easy' in mdtb_conditions:
                # pontine conditions were modelled across difficulty levels, while mdtb conditions were modelled separately
                e = mdtb_conditions.index(eq + 'Easy')
                m = mdtb_conditions.index(eq + 'Med')
                d = mdtb_conditions.index(eq + 'Diff')
                tag_row = tags_mdtb.iloc[[e, m, d], t_idx:]
                tag_row = tag_row.mean(axis=0)

            elif eq not in mdtb_conditions and eq + 'Simple' in mdtb_conditions:
                e = mdtb_conditions.index(eq + 'Simple')
                m = mdtb_conditions.index(eq + 'Seq')
                tag_row = tags_mdtb.iloc[[e, m], t_idx:]
                tag_row = tag_row.mean(axis=0)

            elif eq not in mdtb_conditions and eq + 'ion' in mdtb_conditions:
                e = mdtb_conditions.index(eq + 'ion')
                m = mdtb_conditions.index(eq + 'Viol')
                d = mdtb_conditions.index(eq + 'Scram')
                tag_row = tags_mdtb.iloc[[e, m, d], t_idx:]
                tag_row = tag_row.mean(axis=0)

            elif eq not in mdtb_conditions and eq + '2Back' in mdtb_conditions:
                e = mdtb_conditions.index(eq + '0Back')
                m = mdtb_conditions.index(eq + '2Back')
                tag_row = tags_mdtb.iloc[[e, m], t_idx:]
                tag_row = tag_row.mean(axis=0)

            else:
                tag_row = tags_mdtb.query(
                    'conditionName == @eq').iloc[:, t_idx:].squeeze()
            tag_row = tag_row.append(tags_mdtb_zeros, ignore_index=True)
        elif dset == 'MDTB':
            if cond not in mdtb_conditions:
                cond = mdtb_new2old[cond]
            # print(tags_mdtb.query('conditionName == @cond'))
            tag_row = tags_mdtb.query(
                'conditionName == @cond').iloc[:, t_idx:].squeeze()
            tag_row = tag_row.append(tags_mdtb_zeros, ignore_index=True)
        elif cond in ignore_conditions:
            tag_row = pd.Series([0] * len(all_tags), index=all_tags)
        else:
            tag_rrs = tags_other.query(
                'Condition == @cond & Session == @ses').squeeze()
            tag_after = pd.Series(tags_other.query(
                'Condition == @cond & Session == @ses').iloc[:, -1].tolist()[0], index=all_tags)
            if not isinstance(tag_rrs, pd.Series) and (tag_rrs.iloc[0, :].tags_other == tag_rrs.iloc[1, :].tags_other):
                tag_rrs = tag_rrs.iloc[0, :]
            # Replace tag with tag from tag_rrs
            for i in ['response_selection', 'response_execution', 'saccadic_eye_movement']:
                tag_after[i] = tag_rrs[i].item()
            tag_row = tag_after
        inf = pd.Series([dset, ses, cond], index=[
                        'dataset', 'session', 'condition'])
        full_row = pd.concat([inf, tag_row], axis=0)
        tags.iloc[r, :] = full_row

    # --- Save tags ---
    tags.to_csv(
        f'/Users/callithrix/Documents/Projects/Functional_Fusion/cognitive_ontology/tags.csv', index=False, sep="\t")
    # tags.to_csv(f'{model_dir}/Atlases/tags.csv', index=False, sep="\t")
    pass


def divide_mdtb_by_duration():
    # Get mdtb tags
    tags = pd.read_csv(
        f'{ut.model_dir}/Atlases/Profiles/tags/tags.csv', sep=',')
    tags_mdtb = tags[tags.dataset == 'MDTB']

    # Get mdtb durations
    duration_mdtb = pd.read_csv(
        f'{ut.model_dir}/Atlases/Profiles/tags/mdtb_featureTable.txt', sep='\t')
    # Remove trailing whitespace from condName column
    duration_mdtb['conditionName'] = duration_mdtb['conditionName'].str.strip()

    # For each condition in tags_mdtb, find the corresponding entry in condName column and divide by duration
    for c, cond in enumerate(tags_mdtb.condition):
        # take first entry if there are more than one
        duration = duration_mdtb[duration_mdtb.conditionName ==
                                 cond].duration.iloc[0]
        tags_mdtb.iloc[c, 3:6] = tags_mdtb.iloc[c, 3:6] / duration

    # Replace new tags in tags dataframe
    tags.iloc[tags_mdtb.index, 3:6] = tags_mdtb.iloc[:, 3:6]

    # Save tags
    tags.to_csv(
        f'{ut.model_dir}/Atlases/Profiles/tags/tags_duration.csv', sep=',', index=False)

    pass


if __name__ == "__main__":

    # inspect_cognitive_tags()

    # Load IBC feature tags
    # ibc_features = pd.read_csv(
    #     f'{model_dir}/../../ibc/all_contrasts_corr.csv', sep=","
    # )
    # # get_unique_conditions()
    # get_unique_tags()

    # compile_tags_selftagged_datasets()
    # compile_tags_all_datasets()

    divide_mdtb_by_duration()
