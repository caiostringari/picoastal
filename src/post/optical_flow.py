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
import pandas as pd
import xarray as xr

import cv2

from scipy.interpolate import (LinearNDInterpolator,
                               NearestNDInterpolator,
                               CloughTocher2DInterpolator)

import xarray as xr

from tqdm import tqdm

from matplotlib import path
# from matplotlib import colors as mcolors
import matplotlib.patches as patches
# import matplotlib.pyplot as plt


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
        program_name='Dense Optical Flow (Farneback)',
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

    print("\nComputing optical flow, please wait...\n")

    # Argument parser
    if not gooey:
        parser = argparse.ArgumentParser()
    else:
        parser = GooeyParser(description="Dense Optical Flow (Farneback)")

    # arguments
    if gooey:
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

        parser.add_argument("--mask", "-m",
                    action="store",
                    dest="mask",
                    required=False,
                    default="../../data/flow_mask.geojson",
                    help="Mask as geojson file. Must be a closed polygon.",
                    widget='FileChooser')

        parser.add_argument("--output", "-o",
                            action="store",
                            dest="output",
                            required=False,
                            default="timestack.pkl",
                            help="Output file name (netcdf).",
                            widget='FileSaver')

    else:
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
                            default="flow.nc",
                            help="Output file name (netcdf).")

        parser.add_argument("--number_of_images", "-N",
                            action="store",
                            dest="n_images",
                            default=-1,
                            required=False,
                            help="Number of images to use. Minimum of 2.",)

        parser.add_argument("--mask", "-m",
                            action="store",
                            dest="mask",
                            required=False,
                            default="../../data/flow_mask.geojson",
                            help="Mask as geojson file. Must be a closed polygon.",)

        parser.add_argument("--bbox", "-bbox",
                    action="store",
                    dest="bbox",
                    required=False,
                    default="457237.72,6421856.5,500,500",
                    help="Bounding box to cut the data. Format is "
                            "\'bottom_left,bottom_right,dx,dy\'",)

        parser.add_argument("--epsg",
                            action="store",
                            dest="epsg",
                            required=False,
                            default="28356",
                            help="EPSG code to georefence the output tiff.",)

        parser.add_argument("--method",
                            action="store",
                            dest="interp_method",
                            default="nearest",
                            help="Interpolation method. Default is nearest.")

        parser.add_argument("--dx", "-dx",
                            action="store",
                            dest="dx",
                            default=1,
                            help="Grid resolution (x) in meters. Default is 1m.")

        parser.add_argument("--dy", "-dy",
                            action="store",
                            dest="dy",
                            default=1,
                            help="Grid resolution (y) in meters. Default is 1m.")

    parser.add_argument("--start_time",
                        action="store",
                        dest="start_time",
                        required=False,
                        default="20200101:000000",
                        help="Start time in YYYYMMDD:HHMMSS format. "
                             "Default is \"20200101:000000\"")

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

    parser.add_argument("--pyr_scale",
                        action="store",
                        dest="pyr_scale",
                        default=0.5,
                        help="Parameter specifying the image scale (<1) to build pyramids for each image; "
                             "pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than "
                             "the previous one.")

    parser.add_argument("--levels",
                        action="store",
                        dest="levels",
                        default=3,
                        help="Number of pyramid layers including the initial image; "
                             "levels=1 means that no extra layers are created and only the original images "
                             "are used. ")

    parser.add_argument("--win_size",
                        action="store",
                        dest="win_size",
                        default=3,
                        help="Averaging window size; larger values increase the algorithm "
                             "robustness to image noise and give more chances for fast motion detection, "
                             "but yield more blurred motion field.")

    parser.add_argument("--iterations",
                        action="store",
                        dest="iterations",
                        default=10,
                        help="number of iterations the algorithm does at each pyramid level.")

    parser.add_argument("--poly_n",
                        action="store",
                        dest="poly_n",
                        default=5,
                        help="Size of the pixel neighborhood used to find polynomial expansion in each pixel; "
                             "larger values mean that the image will be approximated with smoother surfaces, "
                             "yielding more robust algorithm and more blurred motion field, "
                             "typically poly_n =5 or 7.")

    parser.add_argument("--poly_sigma",
                        action="store",
                        dest="poly_sigma",
                        default=1.1,
                        help="Standard deviation of the Gaussian that is used to smooth derivatives used as a basis "
                             "for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, "
                             "a good value would be poly_sigma=1.5.")

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
    if int(args.n_images) == -1:
        n_images = len(images)
    else:
        n_images = int(args.n_images)
        if n_images == 1:
            n_images = 2  # need at least 2
        images = images[0:n_images]
    
    print("  -- Processing {} images.".format(n_images))
    first_img = cv2.imread(images[0])

    # read gcp coordinates
    df = pd.read_csv(args.gcps)

    xyz = df[["x", "y", "z"]].values.astype(np.float32)
    uv = df[["u", "v"]].values.astype(np.float32)

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

    # get points inside bbox
    bbox = args.bbox.split(",")
    bbox = np.array([float(bbox[0]), float(bbox[1]),
                     float(bbox[2]), float(bbox[3])])
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             linewidth=2, edgecolor='r', facecolor='none')
    insiders = rect.contains_points(XY)
    insiders_idx = np.arange(0, len(XY), 1)[insiders]
    outsiders_idx = np.arange(0, len(XY), 1)[~insiders]
    # iimg, jimg = np.unravel_index(outsiders_idx, ximg.shape)

    points = XY[insiders_idx, :]

    # define grid
    dx = float(args.dx)
    dy = float(args.dy)
    if dx != dy:
        dx = min(dx, dy)
        dy = min(dx, dy)
        print("   -- warning: can only handle dx=dy. I am using the smallest.")

    xlin = np.arange(bbox[0], bbox[0] + bbox[2], dx)
    ylin = np.arange(bbox[1], bbox[1] + bbox[3], dy)
    grid_x, grid_y = np.meshgrid(xlin, ylin)

    # read the mask
    with open("flow_mask.geojson") as f:
        data = json.load(f)
    coords = np.squeeze(np.array(data["features"][0]["geometry"]["coordinates"]))
    
    # mask points outside the mask
    grid_points = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
    mask = patches.Polygon(coords, linewidth=2, edgecolor='r', facecolor='none')
    insiders = mask.contains_points(grid_points)
    outsiders_idx = np.arange(0, len(grid_points), 1)[~insiders]
    imask, jmask = np.unravel_index(outsiders_idx, grid_x.shape)

    # get new camera matrix
    h,  w = first_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # parameter specifying the image scale (<1) to build pyramids for each image;
    # pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than
    # the previous one.
    pyr_scale = float(args.pyr_scale)   # 0.5

    # number of pyramid layers including the initial image;
    # levels=1 means that no extra layers are created and only the original images are used.
    levels = int(args.levels)  # 3
    
    # averaging window size; larger values increase the algorithm robustness to image noise
    #  and give more chances for fast motion detection, but yield more blurred motion field.
    winsize = int(args.win_size)   #  3
    
    # number of iterations the algorithm does at each pyramid level.
    iterations = int(args.iterations)  # 10
    
    # size of the pixel neighborhood used to find polynomial expansion in each pixel;
    # larger values mean that the image will be approximated with smoother surfaces,
    # yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
    poly_n = int(args.poly_n) # 5

    # standard deviation of the Gaussian that is used to smooth derivatives used as a basis
    # for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7,
    # a good value would be poly_sigma=1.5.
    poly_sigma = float(args.poly_sigma)  # 1.1

    # < timeloop >
    pbar = tqdm(total=len(images) - 1)

    # output variables
    uout = np.zeros([len(images) - 1, grid_x.shape[0], grid_y.shape[1]])
    vout = np.zeros([len(images) - 1, grid_x.shape[0], grid_y.shape[1]])

    times = np.array([start_date] * (len(images) - 1))
    now = start_date
    dt = datetime.timedelta(seconds=1 / freq)

    for i in range(len(images) - 1):

        # read the image
        prv = cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(cv2.imread(images[i + 1]), cv2.COLOR_BGR2GRAY)
        
        # undistort
        prv = cv2.undistort(prv, mtx, dist, None, newcameramtx)
        nxt = cv2.undistort(nxt, mtx, dist, None, newcameramtx)

        # project
        if args.interp_method.lower()  == "linear":
            fp = LinearNDInterpolator(XY[insiders_idx], prv.flatten()[insiders_idx])
            fn = LinearNDInterpolator(XY[insiders_idx], nxt.flatten()[insiders_idx])
        elif args.interp_method.lower()  == "nearest":
            fp = NearestNDInterpolator(XY[insiders_idx], prv.flatten()[insiders_idx])
            fn = NearestNDInterpolator(XY[insiders_idx], nxt.flatten()[insiders_idx])
        elif args.interp_method.lower()  == "ct":
            fp = CloughTocher2DInterpolator(XY[insiders_idx], prv.flatten()[insiders_idx])
            fn = CloughTocher2DInterpolator(XY[insiders_idx], nxt.flatten()[insiders_idx])
        else:
            raise ValueError("Wrong interpolation methd. Use linear, nearest or ct.")
        
        prv = fp(grid_x, grid_y)
        nxt = fn(grid_x, grid_y)

        # compute the flow
        uv = cv2.calcOpticalFlowFarneback(prv, nxt, None, pyr_scale, levels,
                                          winsize, iterations,
                                          poly_n, poly_sigma, 0)

        # convert to m/s
        # magnitude is how much the pixel moved
        mag, ang = cv2.cartToPolar(uv[...,0], uv[...,1])
        displacement = mag * dx  # how much the pixel moved times the grid size
        speed = displacement / (1/freq)  # dS/dt

        # go back to u,v
        u, v = cv2.polarToCart(speed, ang)

        u[imask, jmask] = np.ma.masked  # apply mask
        v[imask, jmask] = np.ma.masked  # apply mask

        uout[i, :, :] = u
        vout[i, :, :] = v

        # time increment
        times[i] = now
        now += dt

        pbar.update()
    pbar.close()

    ds = xr.Dataset()
    # write flow variable
    ds['u'] = (('time', 'x', 'y'), uout)
    ds['v'] = (('time', 'x', 'y'), vout)
    # write coordinates
    ds.coords['time'] = times
    ds.coords["x"] = xlin
    ds.coords["y"] = ylin
    # write to file
    units = 'days since 2000-01-01 00:00:00'
    calendar = 'gregorian'
    encoding = dict(time=dict(units=units, calendar=calendar))
    ds.to_netcdf(args.output, encoding=encoding)

    print("\n Final dataset:")
    print(ds)

    print("\nMy work is done!")


if __name__ == '__main__':
    main()
