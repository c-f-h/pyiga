from pyiga.vis import *

from pyiga import bspline, geometry, approx, hierarchical
import numpy as np

# We don't really "test" the vis functions at the moment but just
# run them to make sure they aren't dead code.

def test_plot_field():
    def f(x, y): return np.sin(x) * np.exp(y)
    geo = geometry.quarter_annulus()
    plot_field(f, physical=True, geo=geo, res=10)
    #
    kvs = 2 * (bspline.make_knots(2, 0.0, 1.0, 5),)
    u = bspline.BSplineFunc(kvs, approx.interpolate(kvs, f))
    plot_field(u, res=10)
    plot_field(u, geo=geo, res=10)

def test_plot_geo():
    plot_geo(geometry.line_segment([0,1], [1,2]))
    plot_geo(geometry.quarter_annulus(), res=10)

def test_animate_field():
    kvs = 2 * (bspline.make_knots(2, 0.0, 1.0, 5),)
    fields = [ bspline.BSplineFunc(kvs,
        approx.interpolate(kvs, lambda x,y: np.sin(t+x) * np.exp(y)))
        for t in range(3) ]
    animate_field(fields, geo=geometry.bspline_quarter_annulus(), res=10)

from .test_hierarchical import create_example_hspace

def test_plot_hierarchical_mesh():
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=1, num_levels=3)
    plot_hierarchical_mesh(hs, levelwise=False)
    plot_hierarchical_mesh(hs, levelwise=True)

def test_plot_hierarchical_cells():
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=1, num_levels=3)
    cells = hs.compute_supports(hs.cell_supp_indices()[-1])
    plot_hierarchical_cells(hs, cells)

def test_plot_hierarchical_cells():
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=1, num_levels=3)
    data = 7.0 * np.arange(hs.total_active_cells)
    plot_active_cells(hs, data)
