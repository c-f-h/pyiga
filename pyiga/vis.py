"""Visualization functions."""
import numpy as np
import matplotlib.pyplot as plt

from . import utils

def plot_field(field, geo=None, res=80, **kwargs):
    kwargs.setdefault('shading', 'gouraud')
    if geo is not None:
        grd = tuple(np.linspace(kv.support()[0], kv.support()[1], res) for kv in geo.kvs)
        XY = utils.grid_eval(geo, grd)
        C = utils.grid_eval(field, grd)
        return plt.pcolormesh(XY[...,0], XY[...,1], C, **kwargs)
    else:
        # assumes that `field` is a BSplineFunc
        grd = tuple(np.linspace(kv.support()[0], kv.support()[1], res) for kv in field.kvs)
        C = utils.grid_eval(field, grd)
        return plt.pcolormesh(grd[1], grd[0], C, **kwargs)


def plot_geo(geo,
             grid=10, gridx=None, gridy=None,
             res=50,
             linewidth=None, color='black'):
    """Plot a wireframe representation of a 2D geometry."""
    if geo.sdim == 1 and geo.dim == 2:
        return plot_curve(geo, res=res, linewidth=linewidth, color=color)
    assert geo.dim == geo.sdim == 2, 'Can only plot 2D geometries'
    if gridx is None: gridx = grid
    if gridy is None: gridy = grid
    supp = geo.support
    gx    = np.linspace(supp[0][0], supp[0][1], gridx)
    meshx = np.linspace(supp[0][0], supp[0][1], res)
    gy    = np.linspace(supp[1][0], supp[1][1], gridy)
    meshy = np.linspace(supp[1][0], supp[1][1], res)

    def plotline(pts, capstyle='butt'):
        plt.plot(pts[:,0], pts[:,1], color=color, linewidth=linewidth,
                solid_joinstyle='round', solid_capstyle=capstyle)

    pts = utils.grid_eval(geo, (gx, meshy))
    plotline(pts[0,:,:], capstyle='round')
    for i in range(1, pts.shape[0]-1):
        plotline(pts[i,:,:])
    plotline(pts[-1,:,:], capstyle='round')

    pts = utils.grid_eval(geo, (meshx, gy))
    plotline(pts[:,0,:], capstyle='round')
    for j in range(1, pts.shape[1]-1):
        plotline(pts[:,j,:])
    plotline(pts[:,-1,:], capstyle='round')


def plot_curve(geo, res=50, linewidth=None, color='black'):
    """Plot a 2D curve."""
    assert geo.dim == 2 and geo.sdim == 1, 'Can only plot 2D curves'
    supp = geo.support
    mesh = np.linspace(supp[0][0], supp[0][1], res)
    pts = utils.grid_eval(geo, (mesh,))
    plt.plot(pts[:,0], pts[:,1], color=color, linewidth=linewidth)
