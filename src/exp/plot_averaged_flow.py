"""
Plot averaged optical flow vectors.

# SCRIPT   : plot_averaged_flow.py
# POURPOSE : Plot averaged optical flow vectors.
# AUTHOR   : Caio Eadi Stringari
# DATE     : 13/07/2021
# VERSION  : 1.0
"""

import argparse

import xarray as xr

from welford import Welford

import cv2

import numpy as np

from tqdm import tqdm

import cmocean as cmo

from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", "-i",
                        action="store",
                        dest="input",
                        default="flow.nc",
                        required=False,
                        help="Input netcdf file.")

    parser.add_argument("--step", "-dx",
                        action="store",
                        dest="step",
                        default=10,
                        required=False,
                        help="Step for plotting vectors.")

    parser.add_argument("--speed_cut", "-cut",
                        action="store",
                        dest="cut",
                        default=2,
                        required=False,
                        help="Maximun speed value.")

    parser.add_argument("--scale", "-qs",
                        action="store",
                        dest="scale",
                        default=100,
                        required=False,
                        help="Scale for the arrows.")

    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        default="flow.png",
                        required=False,
                        help="Output figure name.")

    parser.add_argument("--average", "-a",
                        action="store",
                        dest="average",
                        default="average.tiff",
                        required=False,
                        help="Average image to overlay on.")

    args = parser.parse_args()

    CUT = float(args.cut)
    step = int(args.step)
    scale = float(args.scale)

    # try to read the average image
    try:
        img = plt.imread(args.average)
        has_avg = True
    except Exception:
        has_avg = False

    # read the data
    ds = xr.open_dataset(args.input)
    x = ds["x"].values
    y = ds["y"].values
    grid_x, grid_y = np.meshgrid(x, y)

    # instanciate Welford's objects
    Wd = Welford()
    Wa = Welford()
    # Wd = Welford()

    # compute the average iteratively
    pbar = tqdm(total=len(ds["time"].values))
    for i, time in enumerate(ds["time"].values):

        a = ds.isel(time=i)["angle"].values
        d = ds.isel(time=i)["displacement"].values

        # Wu.add(u)
        Wa.add(a)
        Wd.add(d)

        pbar.update()
    pbar.close()

    # get the average and deviations
    a_mean = Wa.mean
    d_mean = Wd.mean

    u_mean = np.cos(a_mean)  # scaled vector component
    v_mean = np.sin(a_mean)  # scaled vector component

    d_mean = np.ma.masked_equal(d_mean, 0)
    v_mean = np.ma.masked_equal(v_mean, 0)
    u_mean = np.ma.masked_equal(u_mean, 0)

    norm = mcolors.Normalize(vmin=0, vmax=CUT, clip=True)

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))

    m = ax.quiver(grid_x[::step, ::step],
                  grid_y[::step, ::step],
                  u_mean[::step, ::step],
                  v_mean[::step, ::step], d_mean[::step, ::step],
                  cmap="magma", norm=norm, scale=scale)
    
    if has_avg:
        extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
        ax.imshow(img, origin="lower", extent=extent, aspect="equal")
        
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())

    ax.grid()
    ax.set_aspect("equal")

    ax.set_xlabel("x [m]", fontsize=16)
    ax.set_ylabel("y [m]", fontsize=16)

    ax.set_yticklabels(ax.get_yticks(), rotation=90,
                    va="center", fontsize=16)
    ax.set_xticklabels(ax.get_xticks(), rotation=0,
                    ha="center", fontsize=16)

    # colorbar
    lb = "Pixel Displacement [m]"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar = fig.colorbar(m, cax=cax, orientation='vertical')
    cbar.set_label(lb, fontsize=15)

    fig.tight_layout()
    plt.savefig(args.output, dpi=300,
                bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()