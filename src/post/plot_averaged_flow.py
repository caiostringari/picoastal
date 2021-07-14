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

    # instanciate Welford's object
    Wu = Welford()
    Wv = Welford()

    # compute the average iteratively
    pbar = tqdm(total=len(ds["time"].values))
    for i, time in enumerate(ds["time"].values):

        u = ds.isel(time=i)["u"].values
        v = ds.isel(time=i)["v"].values

        Wu.add(u)
        Wv.add(v)

        pbar.update()
    pbar.close()

    # get the average and deviations
    u_mean = Wu.mean
    v_mean = Wv.mean

    u_mean[u_mean > CUT] == np.nan
    v_mean[v_mean > CUT] == np.nan

    # compute the speed
    mag, ang = cv2.cartToPolar(u_mean, v_mean)
    mag[mag > CUT] = CUT
    # mag = np.sqrt(u_mean**2 + v_mean**2)
    
    norm = mcolors.Normalize(vmin=0, vmax=CUT, clip=True)

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))

    m = ax.quiver(grid_x[::step, ::step],
                  grid_y[::step, ::step],
                  u_mean[::step, ::step],
                  v_mean[::step, ::step], mag[::step, ::step],
                  cmap="viridis", norm=norm, scale=scale)
    
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
    lb = "Optically Derived Speed [m/s]"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.05)
    cbar = fig.colorbar(m, cax=cax, orientation='vertical')
    cbar.set_label(lb)

    fig.tight_layout()
    plt.savefig(args.output, dpi=150,
                bbox_inches="tight", pad_inches=0.1)
    plt.show()
    plt.close()


    # # plot
    # fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(12, 7), sharex=True, sharey=True)

    # # plot speed
    # am = ax.pcolormesh(grid_x, grid_y, mag, cmap=cmo.cm.thermal, norm=norm)

    # # plot direction
    # bm = bx.pcolormesh(grid_x, grid_y, ang, cmap=cmo.cm.phase, norm=norm)

    # cm = cx.quiver(grid_x[::step, ::step],
    #               grid_y[::step, ::step],
    #               u_mean[::step, ::step],
    #               v_mean[::step, ::step], mag[::step, ::step],
    #               cmap="viridis", norm=norm)
    
    # if has_avg:
    #     extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    #     ax.imshow(img, origin="lower", extent=extent, aspect="equal")
    #     bx.imshow(img, origin="lower", extent=extent, aspect="equal")
    #     cx.imshow(img, origin="lower", extent=extent, aspect="equal")

    

    # for an in [ax, bx, cx]:
    
    #     an.set_xlim(grid_x.min(), grid_x.max())
    #     an.set_ylim(grid_y.min(), grid_y.max())

    #     an.grid()
    #     ax.set_aspect("equal")

    #     an.set_xlabel("x [m]", fontsize=12)
    #     an.set_ylabel("y [m]", fontsize=12)
    
    #     # an.set_yticklabels(ax.get_yticks(), rotation=0,
    #                     # va="center", fontsize=10)
    #     # an.set_xticklabels(ax.get_xticks(), rotation=90,
    #                     # ha="center", fontsize=10)
    #     # an.xaxis.ticklabel_format(style="sci")
    #     an.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # # colorbar
    # labels = ["Optically Derived Speed [m/s]",
    #           "Optically Derived Phase [rad]",
    #           "Optically Derived Speed [m/s]"]
    # for an, mn, lb in zip([ax, bx, cx], [am, bm, cm], labels):
    #     divider = make_axes_locatable(an)
    #     cax = divider.append_axes('right', size='3%', pad=0.05)
    #     cbar = fig.colorbar(mn, cax=cax, orientation='vertical')
    #     cbar.set_label(lb)

    # fig.tight_layout()
    # plt.savefig(args.output, dpi=150,
    #             bbox_inches="tight", pad_inches=0.1)
    # plt.show()
    # plt.close()