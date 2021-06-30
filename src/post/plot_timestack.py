"""
Plot a timestack created fro timestack.py

# SCRIPT   : timestack.py
# POURPOSE : Plot a timestack created fro timestack.py
# AUTHOR   : Caio Eadi Stringari
# DATE     : 30/06/2021
# VERSION  : 1.0
"""

import pickle

import argparse

import numpy as np

import matplotlib.pyplot as plt


def construct_rgba_vector(img, n_alpha=0):
    """
    Construct RGBA vector to be used to color faces of pcolormesh

    This funciton was taken from Flamingo.
    ----------
    Args:
        img [Mandatory (np.ndarray)]: NxMx3 RGB image matrix

        n_alpha [Mandatory (float)]: Number of border pixels
                                     to use to increase alpha
    ----------
    Returns:
        rgba [Mandatory (np.ndarray)]: (N*M)x4 RGBA image vector
    """

    alpha = np.ones(img.shape[:2])

    if n_alpha > 0:
        for i, a in enumerate(np.linspace(0, 1, n_alpha)):
            alpha[:, [i, -2-i]] = a

    rgb = img[:, :-1, :].reshape((-1, 3))  # we have 1 less faces than grid
    rgba = np.concatenate((rgb, alpha[:, :-1].reshape((-1, 1))), axis=1)

    if np.any(img > 1):
        rgba[:, :3] /= 255.0

    return rgba


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", "-i",
                        action="store",
                        dest="input",
                        required=True,
                        help="Input timestack in pickle format.",)

    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        required=True,
                        help="Output figure name.",)

    args = parser.parse_args()

    # read
    with open(args.input, 'rb') as f:
        inp = pickle.load(f)

    # extrat the needed variables
    rgb = inp["rgb"]
    stack_length = inp["length"]
    stack_time = inp["time"]
    stack_points = inp["points"]

    distance = np.linspace(0, stack_length, stack_points)

    fig, ax = plt.subplots(figsize=(9, 4))

    im = ax.pcolormesh(stack_time, distance, rgb.mean(axis=2)[:-1,:-1], shading="flat")
    im.set_array(None)
    im.set_edgecolor('none')
    im.set_facecolor(construct_rgba_vector(rgb, n_alpha=0))

    ax.set_ylabel("Distance [m]", fontsize=14)
    ax.set_yticklabels(ax.get_yticks(), rotation=90, va="center")


    fig.tight_layout()
    plt.savefig(args.output)
    plt.show()
