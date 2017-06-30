"""Visualization functions."""
import numpy as np
import matplotlib.pyplot as plt

from . import utils

def plot_field(field, geo, res=80, **kwargs):
    grd = tuple(np.linspace(kv.support()[0], kv.support()[1], res) for kv in geo.kvs)
    XY = utils.grid_eval(geo, grd)
    C = utils.grid_eval(field, grd)
    kwargs.setdefault('shading', 'gouraud')
    return plt.pcolormesh(XY[...,0], XY[...,1], C, **kwargs)
