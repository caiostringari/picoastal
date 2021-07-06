"""
Create a timestack from a series of images.

# SCRIPT   : timestack.py
# POURPOSE : Create a timestack from a series of images.
# AUTHOR   : Caio Eadi Stringari
# DATE     : 29/06/2021
# VERSION  : 1.0
"""

import os
import sys

# arguments
import json
import argparse

import datetime

from glob import glob
from natsort import natsorted

import numpy as np

import pickle

import cv2

from scipy.interpolate import griddata

from scipy.spatial import KDTree

from tqdm import tqdm

from matplotlib import path
import matplotlib.patches as patches
import matplotlib.pyplot as plt


try:
    import gooey
    from gooey import GooeyParser
except ImportError:
    gooey = None

import warnings
warnings.simplefilter("ignore", UserWarning)


# <<< GUI >>>
def flex_add_argument(f):
    """Make the add_argument accept (and ignore) the widget option."""

    def f_decorated(*args, **kwargs):
        kwargs.pop('widget', None)
        return f(*args, **kwargs)

    return f_decorated


# monkey-patching a private class
argparse._ActionsContainer.add_argument = flex_add_argument(
    argparse.ArgumentParser.add_argument)


# do not run GUI if it is not available or if command-line arguments are given.
if gooey is None or len(sys.argv) > 1:
    ArgumentParser = argparse.ArgumentParser

    def gui_decorator(f):
        return f
else:
    image_dir = os.path.realpath('../../doc/')
    ArgumentParser = gooey.GooeyParser
    gui_decorator = gooey.Gooey(
        program_name='Timestack Creator',
        default_size=[800, 480],
        navigation="TABBED",
        show_sidebar=True,
        image_dir=image_dir,
        suppress_gooey_flag=True)
# <<< END GUI >>>


def find_homography(uv: np.ndarray, xyz: np.ndarray, mtx: np.ndarray,
                    dist_coeffs: np.ndarray = np.zeros((1, 4)), z: float = 0,
                    compute_error: bool = False):
    """
    Find homography based on ground control points.

    Parameters
    ----------
    uv : np.ndarray
        Nx2 array of image coordinates of gcps.
    xyz : np.ndarray
        Nx3 array of real-world coordinates of gcps.
    mtx : np.ndarray
        3x3 array containing the camera matrix
    dist_coeffs : np.ndarray
        1xN array with distortion coefficients with N = 4, 5 or 8
    z : float
        Real-world elevation to which the image should be projected.
    compute_error : bool
        Will compute re-projection erros in pixels if true.

    Returns
    -------
    error: float
        Rectification error in pixels or nan if compute_error=False.
    H: np.ndarray
        3x3 homography matrix.
    """
    uv = np.asarray(uv).astype(np.float32)
    xyz = np.asarray(xyz).astype(np.float32)
    mtx = np.asarray(mtx).astype(np.float32)

    # compute camera pose
    retval, rvec, tvec = cv2.solvePnP(xyz, uv, mtx, dist_coeffs)

    # convert rotation vector to rotation matrix
    R = cv2.Rodrigues(rvec)[0]

    # assume height of projection plane
    R[:, 2] = R[:, 2] * z

    # add translation vector
    R[:, 2] = R[:, 2] + tvec.flatten()

    # compute homography
    H = np.linalg.inv(np.dot(mtx, R))

    # normalize homography
    H = H / H[-1, -1]

    # compute errors
    if compute_error:
        tot_error = 0
        total_points = 0
        for i in range(len(xyz)):
            reprojected_points, _ = cv2.projectPoints(xyz[i],
                                                      rvec, tvec,
                                                      mtx,
                                                      dist_coeffs)
            tot_error += np.sum(np.abs(uv[i] - reprojected_points)**2)
            total_points += i
        mean_error_px = np.sqrt(tot_error / total_points)
    else:
        mean_error_px = None

    return mean_error_px, H


def rectify_image(img: np.ndarray, mtx: np.ndarray):
    """
    Rectify mage coordinates.

    Parameters
    ----------
    img : np.ndarray
        Input image aray.
    mtx : np.ndarray
        3x3 array containing the camera matrix

    Returns
    -------
    x, y: np.ndarray
        rectified coordinates
    """

    # get_pixel_coordinates(img)
    u, v = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    uv = np.vstack((u.flatten(), v.flatten())).T

    # transform image using homography
    xy = cv2.perspectiveTransform(np.asarray([uv]).astype(np.float32), mtx)[0]

    return xy[:, 0].reshape(u.shape[:2]), xy[:, 1].reshape(v.shape[:2])


@gui_decorator
def main():

    print("\nCreating a timestack, please wait...\n")

    # Argument parser
    if not gooey:
        parser = argparse.ArgumentParser()
    else:
        parser = GooeyParser(description="Timestack Creator")

    # arguments
    if not gooey:
        parser.add_argument("--input", "-i",
                            action="store",
                            dest="input",
                            default="../../data/boomerang",
                            required=False,
                            help="Input folder with images.",)

        parser.add_argument("--camera_matrix", "-mtx",
                            action="store",
                            dest="camera_matrix",
                            default="../../data/flir_tamron_8mm.json",
                            required=False,
                            help="Camera Matrix in JSON or pickle format.",)

        parser.add_argument("--ground_control_points", "-gcps", "--gcps",
                            action="store",
                            dest="gcps",
                            required=False,
                            default="../../data/xyzuv.csv",
                            help="File with x,y,z,u,v data in csv format.",)

        parser.add_argument("--output", "-o",
                            action="store",
                            dest="output",
                            required=False,
                            default="timestack.pkl",
                            help="Timestack in pickle format.")

    else:  # add the same thing but a nicer widget
        parser.add_argument("--input", "-i",
                            action="store",
                            dest="input",
                            default="../../data/boomerang",
                            required=False,
                            help="Input folder with images.",
                            widget='DirChooser')

        parser.add_argument("--camera_matrix", "-mtx",
                            action="store",
                            dest="camera_matrix",
                            default="../../data/flir_tamron_8mm.json",
                            required=False,
                            help="Camera Matrix in JSON or pickle format.",
                            widget='FileChooser')

        parser.add_argument("--ground_control_points", "-gcps", "--gcps",
                            action="store",
                            dest="gcps",
                            required=False,
                            default="../../data/xyzuv.csv",
                            help="File with x,y,z,u,v data in csv format.",
                            widget='FileChooser')

        parser.add_argument("--output", "-o",
                            action="store",
                            dest="output",
                            required=False,
                            default="timestack.pkl",
                            help="Timestack in pickle format.",
                            widget='FileSaver')

    parser.add_argument("--timestack_line",
                        action="store",
                        dest="stackline",
                        required=False,
                        default="457315.2,6422161.5,457599.4,6422063.6",
                        help="Coordinates of the timestack line. Format is"
                             "\'x1,y1,x2,y2\'.")

    parser.add_argument("--start_time",
                        action="store",
                        dest="start_time",
                        required=False,
                        default="20200101:000000",
                        help="Start time in YYYYMMDD:HHMMSS format. "
                             "Default is {20200101:000000}")

    parser.add_argument("--frequency", "-fps",
                        action="store",
                        dest="aquisition_frequency",
                        required=False,
                        default=2,
                        help="Aquistion frequency in Hz. Default is 2Hz.")

    parser.add_argument("--image_format",
                        action="store",
                        dest="image_format",
                        required=False,
                        default="jpg",
                        help="Input images format. Default is jpg.")

    parser.add_argument("--projection_height",
                        action="store",
                        dest="projection_height",
                        required=False,
                        default="-999",
                        help="Project height in meters. Default is -999 which "
                             "uses the mean height of the GCPS.")

    parser.add_argument("--compute_reprojection_error",
                        action="store_true",
                        dest="reprojection_error",
                        help="Compute the re-projection errors.")

    parser.add_argument("--npoints",
                        action="store",
                        dest="npoints",
                        default=1024,
                        help="Number of points in the timestack. "
                             "Default is 1024.")

    parser.add_argument("--neighbours", "-nn",
                        action="store",
                        dest="neighbours",
                        default=1,
                        help="Number of nearest neighbours to consider. "
                             "Default is 1024.")

    parser.add_argument("--statistic",
                        action="store",
                        dest="statistic",
                        default="mean",
                        help="Which statistic to use to compute if neighbours "
                             ">1. Default is np.mean.")

    parser.add_argument("--show_results", "-show",
                        action="store_true",
                        dest="show",
                        help="Show results on screen.")

    args = parser.parse_args()

    # read camera matrix and distortion coefficients
    if args.camera_matrix.lower().endswith("json"):
        with open(args.camera_matrix, 'r') as f:
            cam = json.load(f)
            mtx = np.asarray(cam["camera_matrix"])
            dist = np.asarray(cam["distortion_coefficients"])
    else:
        with open(args.camera_matrix, 'rb') as f:
            cam = pickle.load(f)
            mtx = cam["camera_matrix"]
            dist = cam["distortion_coefficients"]

    # parse time and FPS
    start_date = datetime.datetime.strptime(args.start_time, "%Y%m%d:%H%M%S")
    freq = float(args.aquisition_frequency)

    # search for images
    images = natsorted(glob(args.input + "/*{}".format(args.image_format)))
    start = datetime.datetime.now()
    print(f"  -- Found {len(images)} images, starting at {start}")
    first_img = cv2.imread(images[0])

    # build the timestack line
    npoints = int(args.npoints)
    stackline = args.stackline.split(",")
    stackline = np.array([float(stackline[0]), float(stackline[1]),
                          float(stackline[2]), float(stackline[3])])
    stack_x = np.linspace(stackline[0], stackline[2], npoints)
    stack_y = np.linspace(stackline[1], stackline[3], npoints)
    stack_points = np.vstack([stack_x, stack_y]).T
    stack_length = np.sqrt(
        (stack_x[-1] - stack_x[0])**2 - (stack_y[-1] - stack_y[0])**2)

    # read gcp coordinates
    xyz = []
    uv = []
    f = open(args.gcps, "r")
    for i, line in enumerate(f.readlines()):
        if i > 0:  # ignore header
            xyz.append([line.split(",")[0],
                        line.split(",")[1],
                        line.split(",")[2]])
            uv.append([line.split(",")[3],
                       line.split(",")[4]])
    f.close()
    xyz = np.array(xyz).astype(np.float32)
    uv = np.array(uv).astype(np.float32)

    # rectify
    if int(args.projection_height) == int(-999):
        pheight = xyz[:, 2].mean()
    else:
        pheight = float(args.projection_height)

    error, H = find_homography(uv, xyz, mtx, dist_coeffs=dist, z=pheight,
                               compute_error=args.reprojection_error)
    ximg, yimg = rectify_image(first_img, H)
    if error:
        print(f"  -- Re-projection error is {round(error, 1)} pixels")

    # image coordinate points
    XY = np.vstack([ximg.flatten(), yimg.flatten()]).T

    # build the searching tree
    Tree = KDTree(XY)

    # search for nearest points to the timestack line
    neighbours = int(args.neighbours)

    _, stack_indexes = Tree.query(stack_points, neighbours)
    istk, jstk = np.unravel_index(stack_indexes, ximg.shape)

    if args.statistic == "mean":
        operator = np.mean
    elif args.statistic == "median":
        operator = np.median
    elif args.statistic == "max":
        operator = np.max
    elif args.statistic == "min":
        operator = np.min
    elif args.statistic == "deviation":
        operator = np.std
    elif args.statistic == "variance":
        operator = np.var
    else:
        print("  -- warning: unknown statistic for n. of neighbours > 1, "
              "falling back to np.mean.")
        operator = np.mean

    # < timeloop >

    pbar = tqdm(total=len(images))

    stack_sec = 0
    stack_now = start_date

    rgb_stack = []
    stack_datetimes = []
    stack_seconds = []

    for i, image in enumerate(images):

        # read the image
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        # undistort
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))

        # undistort image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        dst = dst / 255.  # to float

        # extract points
        if neighbours == 1:
            rgb_stack.append(dst[istk, jstk, :])
        else:
            rgb_stack.append(operator(dst[istk, jstk, :], axis=1))

        # time increment
        dt = datetime.timedelta(seconds=1 / freq)
        stack_datetimes.append(stack_now)
        stack_seconds.append(stack_sec)
        stack_now += dt
        stack_sec += 1 / freq

        pbar.update()
    pbar.close()

    # to arrays
    rgb_stack = np.array(rgb_stack)
    rgb_stack = np.swapaxes(rgb_stack, 0, 1)
    stack_times = np.array(stack_datetimes)
    stack_seconds = np.array(stack_seconds)

    # output goes here
    out = {}
    out["seconds"] = stack_seconds
    out["time"] = stack_times
    out["rgb"] = rgb_stack
    out["coordinates"] = stack_points
    out["length"] = stack_length
    out["points"] = npoints
    out["neighbours"] = neighbours
    out["statistic"] = args.statistic
    with open(args.output, 'wb') as f:
        pickle.dump(out, f)

    # plot
    if args.show:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(rgb_stack, extent=[0, stack_length, 0, stack_seconds.max()],
                  origin="lower")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [m]")
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
