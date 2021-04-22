"""
Compute the variance of a series of images.

# SCRIPT   : variance.py
# POURPOSE : Compute the variance of a series of images
# AUTHOR   : Caio Eadi Stringari
# DATE     : 21/04/2021
# VERSION  : 1.0
"""
import argparse
from glob import glob
from natsort import natsorted

import numpy as np

from welford import Welford

from skimage.io import imread, imsave
from skimage.util import img_as_float64

from tqdm import tqdm


if __name__ == "__main__":

    print("\nComputing variance, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument("--input", "-i",
                        action="store",
                        dest="input",
                        required=True,
                        help="Input folder with images file.",)

    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        default="average.png",
                        required=False,
                        help="Output average image name.",)

    args = parser.parse_args()

    # main()

    imlist = natsorted(glob(args.input + "/*"))

    # assuming all images are the same size, get dimensions of first image
    h, w, c = imread(imlist[0]).shape
    N = len(imlist)

    # create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, c), np.float64)

    # instanciate Welford's object
    w = Welford()

    # progress bar
    pbar = tqdm(total=N)

    # add data iteratively
    for i, im in enumerate(imlist):

        # ignore files that are not images
        try:
            img = imread(im)
        except Exception:
            pass

        imarr = img_as_float64(img)

        w.add(imarr)

        pbar.update()

    pbar.close()

    # extract the variance
    arr = w.var_s

    # scale the values and cast to integers
    new_arr = ((arr - arr.min()) *
               (1 / (arr.max() - arr.min()) * 255)).astype('uint8')

    # save the output
    imsave(args.output, new_arr)
