import sys

sys.path.append("..")
import ProbabilisticParcellation.evaluate as ev
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.hierarchical_clustering as cl
import ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas as eva
import ProbabilisticParcellation.scripts.atlas_paper.individual_variability as var
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
        prob = ev.parcel_individual(mname, subject="all", dataset=None, session=None)

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

    # load Uhats
    # prob_a = pt.load(f"{ut.model_dir}/Models/{model_pair[0]}_Uhat.pt")
    # prob_b = pt.load(f"{ut.model_dir}/Models/{model_pair[1]}_Uhat.pt")

    # parcel_a = pt.argmax(prob_a, im=1) + 1
    # parcel_b = pt.argmax(prob_b, dim=1) + 1

    # plt.imshow(np.nanmean(prob_a, axis=1))
    # plt.imshow(np.nanmean(prob_b, axis=1))

    # _, cmap_reordered, labels_reordered = nt.read_lut(lut_dir +
    #                                                   'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered.lut')
    # subject_labels = [f'Subject {i}' for i in range(1, parcel_b.shape[0] + 1)]

    # corr, corr_group = ev.compare_probs(
    #     prob_a, prob_b, method='corr')

    # Save corr as numpy array
    # np.save(f'{ut.model_dir}/Models/{model_pair[0]}_asym_sym_corr.npy', corr)

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

    comp = ev.compare_voxelwise(
        model_pair[0],
        model_pair[1],
        plot=False,
        method="corr",
        save_nifti=True,
        lim=(0, 1),
        individual=False,
    )
    np.save(f"{ut.model_dir}/Models/{model_pair[0]}_asym_sym_corr_group.npy", comp)

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
    pass
