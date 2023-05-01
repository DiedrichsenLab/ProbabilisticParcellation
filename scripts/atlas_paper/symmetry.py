import sys
sys.path.append("..")
import ProbabilisticParcellation.evaluate as ev
import ProbabilisticParcellation.util as ut
import ProbabilisticParcellation.export_atlas as ea
import ProbabilisticParcellation.scripts.atlas_paper.parcel_hierarchy as ph
import ProbabilisticParcellation.scripts.atlas_paper.evaluate_atlas as eva
from Functional_Fusion.dataset import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import torch as pt


def export_uhats(mname='Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered'):
    """Export Uhats for all subjects in a model"""

    prob = ev.parcel_individual(
        mname, subject='all', dataset=None, session=None)

    pt.save(prob, f'{ut.model_dir}/Models/{mname}_Uhat.pt')


if __name__ == "__main__":

    lut_dir = '/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/'
    _, cmap, labels = nt.read_lut(lut_dir +
                                  'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68.lut')
    models = [
        'Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68',
        'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem']

    model_pair = ['Models_03/sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered',
                  'Models_03/asym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_arrange-asym_sep-hem_reordered']

    atlas = 'MNISymC2'

    # load Uhats
    # prob_a = pt.load(f'{ut.model_dir}/Models/{model_pair[0]}_Uhat.pt')
    # prob_b = pt.load(f'{ut.model_dir}/Models/{model_pair[1]}_Uhat.pt')

    # parcel_a = pt.argmax(prob_a, im=1) + 1
    # parcel_b = pt.argmax(prob_b, dim=1) + 1

    # _, cmap_reordered, labels_reordered = nt.read_lut(lut_dir +
    #                                                   'sym_MdPoNiIbWmDeSo_space-MNISymC2_K-68_reordered.lut')
    # subject_labels = [f'Subject {i}' for i in range(1, parcel_b.shape[0] + 1)]

    # corr, corr_group = ev.compare_probs(
    #     prob_a, prob_b, method='corr')

    # Save corr as numpy array
    # np.save(f'{ut.model_dir}/Models/{model_pair[0]}_asym_sym_corr.npy', corr)

    # comp, comp_group = ev.compare_voxelwise(model_pair[0],
    #                                         model_pair[1], plot=False, method='corr', save_nifti=False, lim=(0, 1), individual=True)
    # np.save(
    #     f'{ut.model_dir}/Models/{model_pair[0]}_asym_sym_corr_indiv.npy', comp)

    comp = ev.compare_voxelwise(model_pair[0],
                                model_pair[1], plot=False, method='corr', save_nifti=False, lim=(0, 1), individual=False)
    np.save(
        f'{ut.model_dir}/Models/{model_pair[0]}_asym_sym_corr_group.npy', comp)
    pass
