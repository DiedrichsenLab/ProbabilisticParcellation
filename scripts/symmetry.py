import sys

sys.path.append("..")
import ProbabilisticParcellation.evaluate as ev
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas as eva
import ProbabilisticParcellation.scripts.atlas_paper.individual_variability as var
import ProbabilisticParcellation.learn_fusion_gpu as lf
from Functional_Fusion.dataset import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import torch as pt


def individual_symmetry(mname):
    # Check if Uhats have already been saved. If so, load them
    try:
        prob = pt.load(f"{ut.model_dir}/Models/{mname}_Uhat.pt")
    except:
        prob = ev.parcel_individual(
            mname, subject="all", dataset=None, session=None)

    # Load the model
    fileparts = mname.split("/")
    split_mn = fileparts[-1].split("_")
    info, model = ut.load_batch_best(mname)
    info = ut.recover_info(info, model, mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Load the data
    # for d, dname in enumerate(info.datasets):
    d = 0
    dname = "MDTB"
    data, dinfo, dataset = get_dataset(
        ut.base_dir,
        dname,
        atlas=info.atlas,
        sess=info.sess[d],
        type=info.type[d],
    )

    sym_score = cl.parcel_similarity_indiv(prob, data, sym=False)

    pass


def functional_symmetry(method="model", mname=None):
    if method == "model":
        info, model = ut.load_batch_best(mname)
        # Get winner take-all
        Prob_32 = np.array(model.marginal_prob())
        # Get similarity
        w_cos, _, _ = cl.parcel_similarity(model, plot=False, sym=False)

        # Get the off-diagonal of w_cos: Cross-hemispheric similarity
        indx1 = np.arange(model.K)
        v = np.arange(model.arrange.K)
        indx2 = np.concatenate([v + model.arrange.K, v])
        sym_score = w_cos[indx1, indx2]

    elif method == "data":
        # Get the data
        T = pd.read_csv(ut.base_dir + "/dataset_description.tsv", sep="\t")
        datasets = T.name[:-1].to_numpy()
        sess = np.array(["all"] * len(T), dtype=object)
        data, _, _, subj_ind, _ = lf.build_data_list(
            datasets,
            atlas="MNISymC2",
            type=None,
            join_sess=True,
        )

        suit_atlas, _ = am.get_atlas("MNISymC2", ut.base_dir + "/Atlases")

        # Calculate cosyne similarity for each dataset
        sym_score = []
        indx1 = np.arange(suit_atlas.P)
        v = np.arange(suit_atlas.P // 2)
        indx2 = np.concatenate([v + suit_atlas.P // 2, v])
        # Vall needs to have the shape 514 (for conditions) x 68 (for parcels) x 514 (for conditions) x 68 (for parcels)
        for d, dataset in enumerate(datasets):
            data_dset = data[d]
            data_mirrored = data[d][:, :, suit_atlas.indx_flip]
            # Normalize
            data_dset = data_dset / \
                np.sqrt((data_dset**2).sum(axis=1))[:, None, :]
            data_mirrored = data_mirrored / np.sqrt(
                (data_mirrored**2).sum(axis=1))[:, None, :]
            # Calculate similarity
            corr_dataset = np.zeros(
                (data_dset.shape[0], data_dset.shape[2]))
            for sub in np.arange(data_dset.shape[0]):
                data_sub = data_dset[sub, :, :]
                data_sub_mirrored = data_mirrored[sub, :, :]
                # Calculate cosine similarity
                w_cos_sim = data_sub.T @ data_sub_mirrored
                corr_dataset[sub, :] = np.diagonal(w_cos_sim)
            sym_score.append(corr_dataset)

        # Concatenate all datasets
        sym_score = np.concatenate(sym_score, axis=0)

        np.save(
            f"{ut.model_dir}/Models/Evaluation/nettekoven_68/functional_sim_data.npy", sym_score)

    return sym_score


if __name__ == "__main__":
    # lut_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/'
    # _, cmap, labels = nt.read_lut(lut_dir +
    #                               'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68.lut')
    # models = [
    #     'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68',
    #     'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem']

    # model_pair = ['Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered',
    #               'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem_reordered']

    model_pair = [
        "Models_03/NettekovenSym68_space-MNISymC2",
        "Models_03/NettekovenAsym68_space-MNISymC2",
    ]

    # atlas = 'MNISymC2'

    var.export_uhats(mname=model_pair[0])
    var.export_uhats(mname=model_pair[1])
    var.export_uhats(mname="Models_03/NettekovenAsym32_space-MNISymC2")
    var.export_uhats(mname="Models_03/NettekovenSym32_space-MNISymC2")

    # load Uhats
    # prob_a = pt.load(f"{ut.model_dir}/Models/{model_pair[0]}_Uhat.pt")
    # prob_b = pt.load(f"{ut.model_dir}/Models/{model_pair[1]}_Uhat.pt")

    # parcel_a = pt.argmax(prob_a, dim=1) + 1
    # parcel_b = pt.argmax(prob_b, dim=1) + 1

    # plt.imshow(np.nanmean(prob_a, axis=1))
    # plt.imshow(np.nanmean(prob_b, axis=1))

    # _, cmap_reordered, labels_reordered = nt.read_lut(lut_dir +
    #                                                   'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered.lut')
    # subject_labels = [f'Subject {i}' for i in range(1, parcel_b.shape[0] + 1)]

    # comp, comp_group = ev.compare_voxelwise(
    #     model_pair[0],
    #     model_pair[1],
    #     plot=False,
    #     method="corr",
    #     save_nifti=True,
    #     lim=(0, 1),
    #     individual=True,
    # )
    # np.save(f"{ut.model_dir}/Models/{model_pair[0]}_asym_sym_corr_indiv.npy", comp)

    # comp = ev.compare_voxelwise(
    #     model_pair[0],
    #     model_pair[1],
    #     plot=False,
    #     method="corr",
    #     save_nifti=True,
    #     lim=(0, 1),
    #     individual=False,
    # )
    # np.save(f"{ut.model_dir}/Models/{model_pair[0]}_asym_sym_corr_group.npy", comp)

    #     asym_sym_corr_group = np.load(
    #         f'{ut.model_dir}/Models/{model_pair[0]}_asym_sym_corr_group.npy')
    #     # Replace all values with nans
    #     asym_sym_corr_group = np.ones(asym_sym_corr_group.shape)
    #     # Test if there are any nans
    #     print(np.isnan(asym_sym_corr_group).any())
    #     plt.figure(figsize=(10, 10))
    #     ax = ut.plot_data_flat(asym_sym_corr_group, atlas,
    #                            dtype='func',
    #                            render='matplotlib',
    #                            cmap='hot',
    #                            cscale=(0.8, 1))
    #     plt.show()

    # # Test if there are any zeros
    # print((asym_sym_corr_group == 0).any())

    # individual_symmetry(mname=model_pair[0])

    # sym_score = functional_symmetry(method="model", mname=model_pair[0])
    sym_score = functional_symmetry(method="data")
    np.save(
        f"{ut.model_dir}/Models/{model_pair[0]}_asym_sym_functional_simcorr_data.npy", sym_score)

    pass
