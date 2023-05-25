import numpy as np
from numpy.linalg import eigh, norm
from scipy.linalg import orthogonal_procrustes
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

def get_target_cmap(cmap):
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

def get_target_points(atlas,parcel,sym=False):
    """
    Get target points for MDS  
    We use S1, M3,M4,M1,D3
    """    
    m = np.array([0.68,0.68,0.68])
    colors = np.array([[0.9,0.9,0],[0.9,0.9,0],
                  [0,0.9,0.3],[0,0.9,0.3],
                  [0.1,0.9,0.9],[0.1,0.9,0.9],
                  [0.1,0.1,0.7],[0.1,0.1,0.7],
                  [0.9,0.2,0.9],[0.9,0.2,0.9]])
    points = np.array([[-29,-73,-38],[29,-73,-38],
                       [-18,-53,-19],[18,-53,-19],
                        [-25,-34,-29],[25,-34,-29],
                        [-1,-75,-25],[1,-75,-25],
                       [-36,-60,-30],[36,-60,-30]])
    # Get closest voxel in atlas
    region = np.zeros((points.shape[0],),dtype =int)
    for i,p in enumerate(points):
        d=np.sum((atlas.world-p.reshape(3,1))**2,axis=0)
        region[i]=parcel[np.argmin(d)]-1
    if sym:
        ind = points[:,0]<0
        colors = colors[ind,:]
        region = region[ind]
    return m,region,colors


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
        reg = target[1]
        colors = target[2]

        # Do procrustes-alignment around the mean 
        m=np.mean(W[:,:3],axis=0)
        A = W[reg,:]-m
        B = colors-tm
        R,_ = orthogonal_procrustes(A,B) 
        # Rotate and shift the color space towards the target
        Wm = W @ R + tm 

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

def read_cmap(file):
    """Get matplotlib colour map reads in a saved .cmap file
    Args:
        file (str): Cmap file
    Returns:
        colormap (Listed Colormap):
    """
    cmap = np.loadtxt(file, delimiter=" ", encoding=None)
    cmap = ListedColormap(cmap)
    return cmap

""" Old: eigenvector based colormap: depreciated
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
"""