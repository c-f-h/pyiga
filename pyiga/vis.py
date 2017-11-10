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
