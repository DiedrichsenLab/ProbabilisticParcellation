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
from ProbabilisticParcellation.scripts.parcel_hierarchy import analyze_parcel
from copy import deepcopy

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


def compile_tags():
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


if __name__ == "__main__":

    # inspect_cognitive_tags()

    # Load IBC feature tags
    # ibc_features = pd.read_csv(
    #     f'{model_dir}/../../ibc/all_contrasts_corr.csv', sep=","
    # )
    # # get_unique_conditions()
    # get_unique_tags()

    compile_tags()
