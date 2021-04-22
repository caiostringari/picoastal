"""
Find the brightest and darkest images via color space conversions.

# SCRIPT   : brightest_and_darkest.py
# POURPOSE : ind the brightest and darkest images in a series of images.
# AUTHOR   : Caio Eadi Stringari
# DATE     : 22/04/2021
# VERSION  : 1.0
"""
import argparse
from glob import glob
from natsort import natsorted

import numpy as np

from skimage.io import imread, imsave

from skimage.color import rgb2hsv

from tqdm import tqdm


if __name__ == "__main__":

    print("\nComputing average, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument("--input", "-i",
                        action="store",
                        dest="input",
                        required=True,
                        help="Input folder with images file.",)

    parser.add_argument("--brightest", "-b",
                        action="store",
                        dest="brightest",
                        default="brightest.png",
                        required=False,
                        help="Output name for brightest image.",)

    parser.add_argument("--darkest", "-d",
                        action="store",
                        dest="darkest",
                        default="darkest.png",
                        required=False,
                        help="Output name for darkest image.",)

    args = parser.parse_args()

    # main()

    imlist = natsorted(glob(args.input + "/*"))

    # assuming all images are the same size, get dimensions of first image
    h, w, c = imread(imlist[0]).shape
    N = len(imlist)

    # create a numpy array of floats to store the average (assume RGB images)
    arr = np.zeros((h, w, c), np.float64)

    # progress bar
    pbar = tqdm(total=N)

    # build up average pixel intensities, casting each image as an array of
    # floats
    brightness = []
    for i, im in enumerate(imlist):

        # ignore files that are not images
        try:
            img = imread(im)
        except Exception:
            pass

        imarr = rgb2hsv(img)

        brightness.append(imarr[:, :, 2].sum())

        pbar.update()

    pbar.close()

    # save the outputs
    imsave(args.brightest, imread(imlist[np.argmax(brightness)]))
    imsave(args.darkest, imread(imlist[np.argmin(brightness)]))
