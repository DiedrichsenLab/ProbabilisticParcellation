import sys
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
import nitools as nt
import surfAnalysisPy as surf

atlas_dir = (
    "/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Atlases/"
)
conn_dir = "/Volumes/diedrichsen_data$/data/Cerebellum/connectivity/maps/"
surf_dir = surf.plot._surf_dir


def plot_parcel_summary(parcel="D1", atlas="NettekovenSym32", space="MNISymC2"):
    patlas = nb.load(f"{atlas_dir}{atlas}_space-{space}_probseg.nii")
    _, cmap, labels = nt.read_lut(f"{atlas_dir}{atlas}.lut")
    pseg = suit.flatmap.vol_to_surf(patlas, stats="nanmean", space="MNISymC")

    labels = labels[1:]
    idx = np.array([l.startswith(parcel) for l in labels])
    iidx = np.where(idx)[0]
    p = pseg[:, idx].sum(axis=1)

    fig = plt.figure(figsize=(22, 16))
    spec = fig.add_gridspec(3, 6)

    axC = fig.add_subplot(spec[0:2, 2:4])
    ax = suit.flatmap.plot(
        p,
        cscale=[0, 0.4],
        render="matplotlib",
        cmap="Reds",
        new_figure=False,
        overlay_type="func",
        colorbar=False,
    )

    # Now do connectivity maps
    conn_map = nb.load(conn_dir + "Fusion_L2_05.pscalar.nii")
    weights = nt.cifti.surf_from_cifti(conn_map)
    sc = conn_map.header.get_axis(0).name
    cidx = np.empty((2,), dtype=int)
    for c in range(2):
        cidx[c] = np.where(sc == labels[iidx[c]])[0][0]

    flat = []
    # Use the mirrored flatmap for the left hemisphere
    flat.append(nb.load(surf_dir + "/fs_L/fs_LR.32k.L.flat.surf.gii"))
    flat.append(nb.load(surf_dir + "/fs_R/fs_LR.32k.R.flat.surf.gii"))
    border = []
    border.append(surf_dir + "/fs_L/fs_LR.32k.L.border")
    border.append(surf_dir + "/fs_R/fs_LR.32k.R.border")

    axH = np.empty((2, 2), dtype=object)
    axH[0, 0] = fig.add_subplot(spec[0, 0:2])
    axH[1, 0] = fig.add_subplot(spec[1, 0:2])
    axH[0, 1] = fig.add_subplot(spec[0, 4:])
    axH[1, 1] = fig.add_subplot(spec[1, 4:])

    for h in range(2):
        for c in range(2):
            plt.axes(axH[h, c])
            surf.plot.plotmap(
                weights[h][cidx[c], :],
                flat[h],
                underlay=None,
                overlay_type="func",
                cmap="bwr",
                cscale=[-0.002, 0.002],
                borders=border[h],
            )

    fig.suptitle(parcel)

def plot_parcel_prob(parcel="D1", atlas="NettekovenSym32", 
                     space="MNISymC2",
                     backgroundcolor='w',
                     bordercolor='k'):
    patlas = nb.load(f"{atlas_dir}{atlas}_space-{space}_probseg.nii")
    _, cmap, labels = nt.read_lut(f"{atlas_dir}{atlas}.lut")
    pseg = suit.flatmap.vol_to_surf(patlas, stats="nanmean", space="MNISymC")

    labels = labels[1:]
    idx = np.array([l.startswith(parcel) for l in labels])
    iidx = np.where(idx)[0]
    p = pseg[:, idx].sum(axis=1)

    suit.flatmap.plot(
        p,
        cscale=[0, 0.4],
        render="matplotlib",
        cmap="Reds",
        new_figure=False,
        overlay_type="func",
        colorbar=False,
        backgroundcolor=backgroundcolor,
        bordercolor=bordercolor
    )




if __name__ == "__main__":
    plot_parcel_summary(parcel="M3")
    pass
