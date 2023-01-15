import numpy as np
from numpy.linalg import eigh, norm
import matplotlib.pyplot as plt
from ProbabilisticParcellation.util import *
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from copy import deepcopy


def calc_mds(G,center=False):
    N = G.shape[0]
    if center:
        H = np.eye(N)-np.ones((N,N))/N
        G = H @ G @ H
    G = (G + G.T)/2
    Glam, V = eigh(G)
    Glam = np.flip(Glam,axis=0)
    V = np.flip(V,axis=1)
    W = V[:,:3] * np.sqrt(Glam[:3])

    return W

"""elif type=='hsv':
        Sat=np.sqrt(W[:,0:1]**2)
        Sat = Sat/Sat.max()
        Hue=(np.arctan2(W[:,1],W[:,0])+np.pi)/(2*np.pi)
        Val = (W[:,2]-W[:,2].min())/(W[:,2].max()-W[:,2].min())*0.5+0.4
        rgb = mpl.colors.hsv_to_rgb(np.c_[Hue,Sat,Val])
    elif type=='hsv2':
        Sat=np.sqrt(V[:,0:1]**2)
        Sat = Sat/Sat.max()
        Hue=(np.arctan2(V[:,1],V[:,0])+np.pi)/(2*np.pi)
        Val = (V[:,2]-V[:,2].min())/(V[:,2].max()-V[:,2].min())*0.5+0.4
        rgb = mpl.colors.hsv_to_rgb(np.c_[Hue,Sat,Val])
    elif type=='rgb_cluster':

        rgb = (W-W.min())/(W.max()-W.min()) # Scale between zero and 1
    else:
        raise(NameError(f'Unknown Type: {type}'))
"""

def get_target(cmap):
    if isinstance(cmap,str):
        cmap = mpl.cm.get_cmap(cmap)
    rgb=cmap(np.arange(cmap.N))
    # plot_colormap(rgb)
    tm=np.mean(rgb[:,:3],axis=0)
    A=rgb[:,:3]-tm
    tl,tV=eigh(A.T@A)
    tl = np.flip(tl,axis=0)
    tV = np.flip(tV,axis=1)
    return tm,tl,tV

def make_orthonormal(U):
    """Gram-Schmidt process to make
    matrix orthonormal"""
    n = U.shape[1]
    V=U.copy()
    for i in range(n):
        prev_basis = V[:,0:i]     # orthonormal basis before V[i]
        rem = prev_basis @ prev_basis.T @ U[:,i]
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[:,i] = U[:,i] - rem
        V[:,i] /= norm(V[:,i])
    return V

def plot_colorspace(rgb):
    N,a = rgb.shape
    if a==3:
        rgb = np.c_[rgb,np.ones((N,))]
    rgba = np.r_[rgb,[[0,0,0,1],[1,1,1,1]]]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rgba[:,0],rgba[:,1], rgba[:,2], marker='o',s=70,c=rgba)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    m=np.mean(rgb[:,:3],axis=0)
    A=rgb[:,:3]-m
    l,V=eigh(A.T@A)
    l = np.flip(l,axis=0)
    V = np.flip(V,axis=1)

    B = V * np.sqrt(l) * 0.5
    for i in range(2):
        ax.quiver(m[0],m[1],m[2],B[0,i],B[1,i],B[2,i])
    return m,l,V


def colormap_mds(W,target=None,clusters=None,gamma=0.3):
    """Map the similarity structure of MDS to a colormap
    Args:
        W (ndarray): N x 3 array of original multidimensional scaling
        target (tuple): Target origin [0] directions[1] of the desired map
        clusters (ndarray): distorts color towards cluster mean
        gamma (float): Strength of cluster mean
    Returns:
        colormap (Listed Colormap):
    """
    N = W.shape[0]
    if target is not None:
        tm=target[0]
        tV = target[1]

        # Get the eigenvalues of W around the origin.
        m=np.mean(W[:,:3],axis=0)
        A=W-m
        # Get the eigenvalues in ascending order
        l,V=eigh(A.T@A)
        l = np.flip(l,axis=0)
        V = np.flip(V,axis=1)
        # Rotate and shift the color space towards the target
        Wm = A @ V @ tV.T
        Wm += tm
    # rgb = (W-W.min())/(W.max()-W.min()) # Scale between zero and 1
    Wm[Wm<0]=0
    Wm[Wm>1]=1
    if clusters is not None:
        M = np.zeros((clusters.max(),3))
        for i in np.unique(clusters):
            M[i-1,:]=np.mean(Wm[clusters==i,:],axis=0)
            Wm[clusters==i,:]=(1-gamma) * Wm[clusters==i,:] + gamma * M[i-1]

    colors = np.c_[Wm,np.ones((N,))]
    colorsp = np.r_[np.zeros((1,4)),colors] # Add empty for the zero's color
    newcmp = ListedColormap(colorsp)
    return newcmp
