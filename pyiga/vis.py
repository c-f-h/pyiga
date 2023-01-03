"""Visualization functions."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from . import utils

def plot_field(field, geo=None, res=80, physical=False, **kwargs):
    """Plot a scalar field, optionally over a geometry."""
    kwargs.setdefault('shading', 'gouraud')
    if np.isscalar(res):
        res = (res, res)
    if geo is not None:
        grd = tuple(np.linspace(s[0], s[1], r) for (s,r) in zip(geo.support, res))
        XY = utils.grid_eval(geo, grd)
        if physical:
            C = utils.grid_eval_transformed(field, grd, geo)
        else:
            C = utils.grid_eval(field, grd)
        return plt.pcolormesh(XY[...,0], XY[...,1], C, **kwargs)
    else:
        # assumes that `field` is a BSplineFunc or equivalent
        grd = tuple(np.linspace(s[0], s[1], r) for (s,r) in zip(field.support, res))
        C = utils.grid_eval(field, grd)
        return plt.pcolormesh(grd[1], grd[0], C, **kwargs)


def plot_geo(geo,
             grid=10, gridx=None, gridy=None, gridz=None,
             res=50,
             linewidth=None, color='black'):
    """Plot a wireframe representation of a 2D geometry."""
    assert (geo.dim == 2 or geo.dim == 3), 'Can only represent geometries in 2D or 3D!' 
    if geo.sdim == 1:
        return plot_curve(geo, res=res, linewidth=linewidth, color=color)
    if geo.sdim == 2:
        return plot_surface(geo, grid=grid, gridx=gridx, gridy=gridy, res=res, linewidth=linewidth, color=color)
    else:
        if gridx is None: gridx = grid
        if gridy is None: gridy = grid
        if gridz is None: gridz = grid
    supp = geo.support

    # if gridx/gridy is not an array, build an array with given number of ticks
    if np.isscalar(gridx):
        gridx = np.linspace(supp[0][0], supp[0][1], gridx)
    if np.isscalar(gridy):
        gridy = np.linspace(supp[1][0], supp[1][1], gridy)
    if np.isscalar(gridz):
        gridz = np.linspace(supp[2][0], supp[2][1], gridz)

    meshx = np.linspace(supp[0][0], supp[0][1], res)
    meshy = np.linspace(supp[1][0], supp[1][1], res)
    meshz = np.linspace(supp[2][0], supp[2][1], res)
    
    def plotline(pts, capstyle='butt', color=color):
        plt.plot(pts[:,0], pts[:,1], pts[:,2], color=color, linewidth=linewidth,
            solid_joinstyle='round', solid_capstyle=capstyle)

        
    pts = utils.grid_eval(geo, (gridx, gridy, meshz))
    for i in range(0, pts.shape[0]):
        for j in range(0, pts.shape[1]):
            if i==0 or i == pts.shape[0]-1 or j==0 or j== pts.shape[1]-1:
                plotline(pts[i,j,:,:], capstyle='round')
            else:
                plotline(pts[i,j,:,:])
    
    pts = utils.grid_eval(geo, (gridx, meshy, gridz))
    for i in range(0, pts.shape[2]):
        for j in range(0, pts.shape[0]):
            if i==0 or i == pts.shape[2]-1 or j==0 or j== pts.shape[0]-1:
                plotline(pts[j,:,i,:], capstyle='round')
            else:
                plotline(pts[j,:,i,:])

    pts = utils.grid_eval(geo, (meshx, gridy, gridz))
    for i in range(0, pts.shape[1]):
        for j in range(0, pts.shape[2]):
            if i==0 or i == pts.shape[1]-1 or j==0 or j== pts.shape[2]-1:
                plotline(pts[:,i,j,:], capstyle='round')
            else:
                plotline(pts[:,i,j,:])
    
def plot_surface(geo,
                 grid=10, gridx=None, gridy=None,
                 res=50,
                 linewidth=None, color='black'):
    """Plot a 2D or 3D surface."""
    assert geo.sdim == 2 and (geo.dim == 2 or geo.dim == 3), "Can only plot surfaces."
    if gridx is None: gridx = grid
    if gridy is None: gridy = grid
    supp = geo.support

    # if gridx/gridy is not an array, build an array with given number of ticks
    if np.isscalar(gridx):
        gridx = np.linspace(supp[0][0], supp[0][1], gridx)
    if np.isscalar(gridy):
        gridy = np.linspace(supp[1][0], supp[1][1], gridy)

    meshx = np.linspace(supp[0][0], supp[0][1], res)
    meshy = np.linspace(supp[1][0], supp[1][1], res)

    def plotline(pts, capstyle='butt', color=color):
        if geo.dim == 3:
            plt.plot(pts[:,0], pts[:,1], pts[:,2], color=color, linewidth=linewidth,
                                       solid_joinstyle='round', solid_capstyle=capstyle)
        if geo.dim == 2:
            plt.plot(pts[:,0], pts[:,1], color=color, linewidth=linewidth,
                solid_joinstyle='round', solid_capstyle=capstyle)

    pts = utils.grid_eval(geo, (gridx, meshy))
    plotline(pts[0,:,:], capstyle='round')
    for i in range(1, pts.shape[0]-1):
        plotline(pts[i,:,:])
    plotline(pts[-1,:,:], capstyle='round')

    pts = utils.grid_eval(geo, (meshx, gridy))
    plotline(pts[:,0,:], capstyle='round')
    for j in range(1, pts.shape[1]-1):
        plotline(pts[:,j,:])
    plotline(pts[:,-1,:], capstyle='round')


def plot_curve(geo, res=50, linewidth=None, color='black'):
    """Plot a 2D curve."""
    assert (geo.dim == 2 or geo.dim == 3) and geo.sdim == 1, 'Can only plot 2D curves'
    supp = geo.support
    mesh = np.linspace(supp[0][0], supp[0][1], res)
    pts = utils.grid_eval(geo, (mesh,))
    if geo.dim == 3:
        plt.axes(projection='3d').plot3D(pts[:,0], pts[:,1], pts[:,2], color=color, linewidth=linewidth)
    if geo.dim == 2:
        plt.plot(pts[:,0], pts[:,1], color=color, linewidth=linewidth)


def animate_field(fields, geo, vrange=None, res=(50,50), cmap=None, interval=50, progress=False):
    """Animate a sequence of scalar fields over a geometry."""
    fields = list(fields)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    if np.isscalar(res):
        res = (res, res)
    grd = tuple(np.linspace(s[0], s[1], r) for (s,r) in zip(geo.support, res))
    XY = geo.grid_eval(grd)
    C = np.zeros(res)

    if vrange is None:
        # determine range of values from first field
        C = utils.grid_eval(fields[0], grd)
        vrange = (C.min(), C.max())

    quadmesh = plt.pcolormesh(XY[...,0], XY[...,1], C, shading='gouraud', cmap=cmap,
                vmin=vrange[0], vmax=vrange[1], axes=ax)
    fig.colorbar(quadmesh, ax=ax)

    tqdm = utils.progress_bar(progress)
    pbar = tqdm(total=len(fields))
    def anim_func(i):
        C = utils.grid_eval(fields[i], grd)
        quadmesh.set_array(C.ravel())
        pbar.update()
        if i == len(fields) - 1:
            pbar.close()

    return animation.FuncAnimation(fig, anim_func, frames=len(fields), interval=interval)

class HSpaceVis:
    def __init__(self, hspace):
        assert hspace.dim == 2, 'Only 2D visualization implemented'
        self.hspace = hspace

    @staticmethod
    def vis_rect(r):
        Y, X = r        # note: last axis = x
        return matplotlib.patches.Rectangle((X[0], Y[0]), X[1]-X[0], Y[1]-Y[0])

    def cell_to_rect(self, lv, c):
        return self.vis_rect(self.hspace.cell_extents(lv, c))

    def setup_axes(self):
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    def plot_level(self, lv, color_act='steelblue', color_deact='lavender'):
        ax = self.setup_axes()

        from matplotlib.collections import PatchCollection
        if color_act is not None:
            Ra = [self.cell_to_rect(lv, c) for c in self.hspace.active_cells(lv)]
            ax.add_collection(PatchCollection(Ra, facecolor=color_act, edgecolor='black'))
        if color_deact is not None:
            Rd = [self.cell_to_rect(lv, c) for c in self.hspace.deactivated_cells(lv)]
            ax.add_collection(PatchCollection(Rd, facecolor=color_deact, edgecolor='black'));

    def plot_level_cells(self, cells, lv, color_act='steelblue', color_deact='white'):
        ax = self.setup_axes()

        from matplotlib.collections import PatchCollection
        if color_act is not None:
            Ra = [self.cell_to_rect(lv, c) for c in self.hspace.active_cells(lv) if c in cells]
            ax.add_collection(PatchCollection(Ra, facecolor=color_act, edgecolor='black'))
        if color_deact is not None:
            Rd = [self.cell_to_rect(lv, c) for c in self.hspace.active_cells(lv) if c not in cells]
            ax.add_collection(PatchCollection(Rd, facecolor=color_deact, edgecolor='black'))

    def plot_active_cells(self, values, cmap=None, edgecolor=None):
        ax = self.setup_axes()

        from matplotlib.collections import PatchCollection
        act_cells = self.hspace.active_cells(flat=True)
        if not len(values) == len(act_cells):
            raise ValueError('invalid length of `values` array')
        R = [self.cell_to_rect(lv, c) for (lv, c) in act_cells]
        p = PatchCollection(R, cmap=cmap, edgecolor=edgecolor)
        p.set_array(values)
        ax.add_collection(p)
        return ax, p

    def vis_function(self, lv, jj):
        r = self.vis_rect(self.hspace.function_support(lv, jj))
        r.set_fill(False)
        r.set_edgecolor('red')
        r.set_linewidth(3)
        return r

def plot_hierarchical_mesh(hspace, levels='all', levelwise=False, color_act='steelblue', color_deact='lavender'):
    """Visualize the mesh of a 2D hierarchical spline space.

    Args:
        hspace (:class:`.HSpace`): the space to be plotted
        levels: either 'all' or a list of levels to plot
        levelwise (bool): if True, show each level (including active and deactivated
            basis functions) in a separate subplot
        color_act: the color to use for the active cells
        color_deact: the color to use for the deactivated cells (only shown if `levelwise` is True)
    """
    V = HSpaceVis(hspace)
    if levels == 'all':
        levels = tuple(range(hspace.numlevels))
    else:
        levels = tuple(levels)

    for j,lv in enumerate(levels):
        if levelwise:
            plt.subplot(1, len(levels), j+1)
        V.plot_level(lv, color_act=color_act, color_deact=color_deact if levelwise else None)

def plot_hierarchical_cells(hspace, cells, color_act='steelblue', color_deact='white'):
    """Visualize cells of a 2D hierarchical spline space.

    Args:
        hspace (:class:`.HSpace`): the space to be plotted
        cells: dict of sets of selected active cells
        color_act: the color to use for the selected cells
        color_deact: the color to use for the remaining cells
    """
    V = HSpaceVis(hspace)

    for lv in range(hspace.numlevels):
        V.plot_level_cells(cells.get(lv, {}), lv, color_act=color_act, color_deact=color_deact)

def plot_active_cells(hspace, values, cmap=None, edgecolor=None):
    """Plot the mesh of active cells with colors chosen according to the given
    `values`."""
    return HSpaceVis(hspace).plot_active_cells(values, cmap=cmap)
