"""Visualization functions."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from . import utils

def plot_field(field, geo=None, res=80, physical=False, contour=False, **kwargs):
    """Plot a scalar field, optionally over a geometry."""
    kwargs.setdefault('shading', 'gouraud')
    #assert field.dim==1
    if np.isscalar(res):
        res = (res, res)
    if geo is not None:
        grd = tuple(np.linspace(s[0], s[1], r) for (s,r) in zip(geo.support, res))
        XY = utils.grid_eval(geo, grd)
        if physical:
            C = utils.grid_eval_transformed(field, grd, geo)
        else:
            C = utils.grid_eval(field, grd)
        if contour:
            return plt.contourf(XY[...,0],XY[...,1], C, **kwargs)
        else:
            return plt.pcolormesh(XY[...,0], XY[...,1], C, **kwargs)
    else:
        # assumes that `field` is a BSplineFunc or equivalent
        grd = tuple(np.linspace(s[0], s[1], r) for (s,r) in zip(field.support, res))
        C = utils.grid_eval(field, grd)
        if contour:
            plt.contourf(grd[1],grd[0], C, **kwargs)
        else:
            return plt.pcolormesh(grd[1], grd[0], C, **kwargs)
    
#TODO
# def plot_mp_field(fields, MP, res=80, physical=False, contour=False, **kargs):
#     """Plot a scalar field, optionally over a Multi-patch geometry."""
#     kwargs.setdefault('shading', 'gouraud')
#     #assert field.dim==1
#     if np.isscalar(res):
#         res = (res, res)
        
def plot_quiver(field, geo=None, res=10, physical=False, **kwargs):
    """plot a vector field, optionally over a geometry."""
    #assert field.dim==2
    if np.isscalar(res):
        res = (res, res)
    if geo is not None:
        grd = tuple(np.linspace(s[0], s[1], r) for (s,r) in zip(geo.support, res))
        XY = utils.grid_eval(geo, grd)
        if physical:
            C = utils.grid_eval_transformed(field, grd, geo)
        else:
            C = utils.grid_eval(field, grd)
            return plt.quiver(XY[...,0], XY[...,1],C[:,:,0], C[:,:,1],**kwargs)
            #return plt.pcolormesh(XY[...,0], XY[...,1], C, **kwargs)
    else:
        # assumes that `field` is a BSplineFunc or equivalent
        grd = tuple(np.linspace(s[0], s[1], r) for (s,r) in zip(field.support, res))
        C = utils.grid_eval(field, grd)
        return plt.quiver(grd[1], grd[0], C[:,:,0], C[:,:,1], **kwargs)
    
def plot_mp_quiver():
    print(1)
        
def plot_geo(geo,
             grid=10, gridx=None, gridy=None, gridz=None,
             res=50,
             linewidth=None, lcolor='black', color=None, boundary=True, bcolor='black', controlgrid=False, zorder=1):
    """Plot a wireframe representation of a geometry."""
    assert (geo.dim == 2 or geo.dim == 3), 'Can only represent geometries in 2D or 3D!' 
    if geo.sdim == 1:
        return plot_curve(geo, res=res, linewidth=linewidth, lcolor=color, controlgrid=controlgrid, zorder=zorder)
    if geo.sdim == 2:
        return plot_surface(geo, grid=grid, gridx=gridx, gridy=gridy, res=res, linewidth=linewidth, lcolor=lcolor, color=color, boundary=boundary, bcolor=bcolor, controlgrid=controlgrid, zorder=zorder)
    else:
        if gridx is None: gridx = grid
        if gridy is None: gridy = grid
        if gridz is None: gridz = grid
    if geo._support_override is not None:
        supp = geo._support_override
    else:
        supp = geo.support

    # if gridx/gridy is not an array, build an array with given number of ticks
    if np.isscalar(gridx):
        gridx = np.linspace(supp[0][0], supp[0][1], max(gridx,2))
    if np.isscalar(gridy):
        gridy = np.linspace(supp[1][0], supp[1][1], max(gridy,2))
    if np.isscalar(gridz):
        gridz = np.linspace(supp[2][0], supp[2][1], max(gridz,2))

    meshx = np.linspace(supp[0][0], supp[0][1], res)
    meshy = np.linspace(supp[1][0], supp[1][1], res)
    meshz = np.linspace(supp[2][0], supp[2][1], res)
    
    def plotline(pts, capstyle='butt',color=lcolor):
        plt.plot(pts[:,0], pts[:,1], pts[:,2] , color=lcolor, linewidth=linewidth,
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
                 linewidth=None, lcolor='black', color=None, boundary=True, bcolor='black', controlgrid = False, zorder=1):
    """Plot a 2D or 3D surface."""
    assert geo.sdim == 2 and (geo.dim == 2 or geo.dim == 3), "Can only plot surfaces."
    if gridx is None: gridx = grid
    if gridy is None: gridy = grid
    if geo._support_override is not None:
        supp = geo._support_override
    else:
        supp = geo.support

    # if gridx/gridy is not an array, build an array with given number of ticks
    if np.isscalar(gridx):
        gridx = np.linspace(supp[1][0], supp[1][1], max(gridx,2))
    if np.isscalar(gridy):
        gridy = np.linspace(supp[0][0], supp[0][1], max(gridy,2))

    meshx = np.linspace(supp[1][0], supp[1][1], res)
    meshy = np.linspace(supp[0][0], supp[0][1], res)
    
    def plotline(pts, capstyle='butt', color=color, linewidth=None, zorder=1):
        if geo.dim == 3:
            plt.plot(pts[:,0], pts[:,1], pts[:,2], color=color, linewidth=linewidth,
                                       solid_joinstyle='round', solid_capstyle=capstyle, zorder=zorder)
        if geo.dim == 2:
            plt.plot(pts[:,0], pts[:,1], color=color, linewidth=linewidth,
                solid_joinstyle='round', solid_capstyle=capstyle, zorder = zorder)

    #print(meshy)
    pts1 = utils.grid_eval(geo, (gridy, meshx))
    pts2 = utils.grid_eval(geo, (meshy, gridx))

    if controlgrid:
        ctr = geo.control_points()
        plt.scatter(ctr[:,:,0],ctr[:,:,1], c="red",zorder=100+zorder)
        plt.plot(ctr[:,:,0],ctr[:,:,1], c="red",zorder=10)+zorder;
        ctr=np.transpose(ctr,(1,0,2));
        plt.plot(ctr[:,:,0],ctr[:,:,1], c="red",zorder=10+zorder);

    if boundary:
        plotline(pts1[0,:,:], capstyle='round', linewidth=linewidth, color=bcolor, zorder=1000+zorder)
        plotline(pts1[-1,:,:], capstyle='round', linewidth=linewidth, color=bcolor,zorder=1000+zorder)
        plotline(pts2[:,0,:], capstyle='round', linewidth=linewidth, color=bcolor, zorder=1000+zorder)
        plotline(pts2[:,-1,:], capstyle='round', linewidth=linewidth, color=bcolor,zorder=1000+zorder)
    
    for i in range(1, pts1.shape[0]-1):
        plotline(pts1[i,:,:], linewidth=linewidth, color=lcolor)

    for j in range(1, pts2.shape[1]-1):
        plotline(pts2[:,j,:], linewidth=linewidth, color=lcolor)
    
    if color:
        bpts0 = utils.grid_eval(geo,(meshy, np.linspace(supp[1][0], supp[1][1], 2)))
        bpts1 = utils.grid_eval(geo, (np.linspace(supp[0][0], supp[0][1], 2),meshx))
        x=np.concatenate([bpts1[0,:,0],bpts0[:,1,0],np.flip(bpts1[1,:,0]),np.flip(bpts0[:,0,0])])
        y=np.concatenate([bpts1[0,:,1],bpts0[:,1,1],np.flip(bpts1[1,:,1]),np.flip(bpts0[:,0,1])])
        plt.fill(x,y,color=color)

def plot_curve(geo, res=50, linewidth=None, lcolor='black', controlgrid=False, zorder=1):
    """Plot a 2D curve."""
    if not lcolor:
        lcolor='black'
    assert (geo.dim == 2 or geo.dim == 3) and geo.sdim == 1, 'Can only plot 2D curves'
    if geo._support_override:
        supp = geo._support_override
    else:
        supp = geo.support
    mesh = np.linspace(supp[0][0], supp[0][1], res)
    pts = utils.grid_eval(geo, (mesh,))
    if controlgrid:
        ctr = geo.control_points()
    if geo.dim == 3:
        if controlgrid:
            plt.axes(projection='3d').scatter(ctr[:,0], ctr[:,1], ctr[:,2], color="red", zorder=100+zorder)
            plt.axes(projection='3d').plot3D(ctr[:,0], ctr[:,1], ctr[:,2], color="red", linewidth=linewidth, zorder=10+zorder)
        plt.axes(projection='3d').plot3D(pts[:,0], pts[:,1], pts[:,2], color=lcolor, linewidth=linewidth, zorder=zorder)
    if geo.dim == 2:
        if controlgrid:
            plt.scatter(ctr[:,0], ctr[:,1], color="red", zorder=100+zorder)
            plt.plot(ctr[:,0], ctr[:,1], color='red',  linewidth=linewidth, zorder=10+zorder)
        plt.plot(pts[:,0], pts[:,1], color=lcolor, linewidth=linewidth, zorder=zorder)


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

def spy(A, marker=None, markersize=10, cbar=False, cmap = plt.cm.jet, **kwargs):
    fig = plt.figure(figsize=kwargs.get('figsize'))
    ax = plt.axes()
    
    A = A.tocoo()
    data, I, J = A.data, A.row, A.col
    if cbar:
        plt.scatter(J, I, c=data, marker="s", s=markersize, cmap = cmap)
    else:
        plt.scatter(J, I, s=markersize, marker="s")
    ax.axis('scaled');
    ax.set_xlim(-0.5,A.shape[1]-0.5)
    ax.set_ylim(-0.5,A.shape[0]-0.5)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    if cbar:
        cax = fig.add_axes([ax.get_position().x1+0.03,ax.get_position().y0,0.02,ax.get_position().height])
        plt.colorbar(cax=cax) # Similar to fig.colorbar(im, cax = cax)
    
    
