"""
Plot averaged optical flow vectors

# SCRIPT   : plot_averaged_flow.py
# POURPOSE : Create a timestack from a series of images.
# AUTHOR   : Caio Eadi Stringari
# DATE     : 09/07/2021
# VERSION  : 1.0
"""

import xarray as xr

from welford import Welford

import cv2

import numpy as np

from tqdm import tqdm

from matplotlib import path
from matplotlib import colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def scale_array(x, srcRange, dstRange):
    """Scale a variable to a new range."""
    return ((x - srcRange[0]) * (dstRange[1] - dstRange[0]) /
            (srcRange[1] - srcRange[0]) + dstRange[0])


if __name__ == '__main__':

    dt = (1/2) # 1 over sample rate

    img = plt.imread("average.tiff")

    ds = xr.open_dataset("flow.nc")
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

    u_mean[u_mean > 2] == 0
    v_mean[v_mean > 2] == 0

    u_mean[np.where(img[...,1] == 0)] = 0
    v_mean[np.where(img[...,1] == 0)] = 0

    # magnitude is how much the pixel moved
    mag, ang = cv2.cartToPolar(u_mean, v_mean)
    mag /= dt  # this is the velocity

    # ones = np.ones(ang.shape).astype(type(mag[0][0]))
    # unorm, vnorm = cv2.polarToCart(ones, ang)
    #
    # magnorm = scale_array(mag, (mag.min(), mag.max()), (0, 1))
    norm = mcolors.Normalize(vmin=0, vmax=2., clip=True)

    # plot
    step = 10
    fig, ax = plt.subplots(figsize=(10, 10))

    m = ax.quiver(grid_x[::step, ::step],
                  grid_y[::step, ::step],
                  u_mean[::step, ::step],
                  v_mean[::step, ::step], mag[::step, ::step],
                  cmap="Greys", norm=norm)

    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    ax.imshow(img, origin="lower", extent=extent, aspect="equal")

    plt.colorbar(mappable=m, cax=None, ax=ax)

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

    fig.tight_layout()
    plt.show()
    # break
    # plt.savefig("flow/{}.png".format(str(i).zfill(6)), dpi=120,
    #             bbox_inches="tight", pad_inches=0.1)
    # plt.close()
