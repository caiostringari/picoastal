"""
Rectify a given image.

# SCRIPT   : rectify.py
# POURPOSE : Rectify a given image.
# AUTHOR   : Caio Eadi Stringari
# DATE     : 29/06/2021
# VERSION  : 1.0
"""

import os
import sys

# arguments
import json
import argparse

import numpy as np

import pickle

import cv2

from scipy.interpolate import griddata

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from osgeo import gdal
from osgeo import osr

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
        program_name='Image Rectification',
        default_size=[800, 480],
        navigation="TABBED",
        show_sidebar=True,
        image_dir=image_dir,
        suppress_gooey_flag=True,
    )
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


def save_as_geotiff(grid_x: np.ndarray, grid_y: np.ndarray, dx: float,
                    dy: float, rgb: np.ndarray, epsg: int, outfile: str):
    """
    Save output image as geotiff using GDAL.

    Parameters
    ----------
    grid_x : np.ndarray
        Grid x-coordinates.
    grid_y : np.ndarray
        Grid y-coordinates.
    dx, dy : float
        Grid resolution in x and y.
    rgb : np.ndarray
        Image data.
    epsg : int
        EPSG code for georefencing.
    outfile : str
        Output file name.

    Returns
    -------
    None
        Will write to file instead.
    """
    # set geotransform
    nx = rgb.shape[0]
    ny = rgb.shape[1]
    geotransform = [grid_x.min(), dx, 0, grid_y.min(), 0, dy]

    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(
        outfile, ny, nx, 3, gdal.GDT_Byte)

    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    # EPSG:28356 - GDA94 / MGA zone 56 - Projected
    srs.ImportFromEPSG(int(epsg))
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file

    # write rgb bands to the raster
    dst_ds.GetRasterBand(1).WriteArray(rgb[:, :, 0])
    dst_ds.GetRasterBand(2).WriteArray(rgb[:, :, 1])
    dst_ds.GetRasterBand(3).WriteArray(rgb[:, :, 2])

    # write to disk
    dst_ds.FlushCache()
    dst_ds = None


def plot(grid_x: np.ndarray, grid_y: np.ndarray, rgb: np.ndarray,
         gcps: np.ndarray = None):
    """
    Plot the rectification results

    Parameters
    ----------
    grid_x : np.ndarray
        Grid x-coordinates.
    grid_y : np.ndarray
        Grid y-coordinates.
    rgb : np.ndarray
        Image data.
    gcps : np.ndarray
        Nx3 array with GCP coordinates. Optional.

    Returns
    -------
    None
        Will show plot on screen.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    extent = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    ax.imshow(rgb, origin="lower", extent=extent, aspect="equal")

    if len(gcps) > 0:
        ax.scatter(gcps[:, 0], gcps[:, 1], color="r", lw=2, marker="+", s=50,
                   label="GCPs")

    ax.legend(fontsize=16)

    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())

    ax.grid()
    ax.set_aspect("equal")

    ax.set_xlabel("x [m]", fontsize=16)
    ax.set_ylabel("y [m]", fontsize=16)
    ax.set_yticklabels(ax.get_yticks(), rotation=90, va="center", fontsize=16)
    ax.set_xticklabels(ax.get_xticks(), rotation=0, ha="center", fontsize=16)

    fig.tight_layout()
    plt.show()


@gui_decorator
def main():

    print("\nCreating image geometry, please wait...\n")

    # Argument parser
    if not gooey:
        parser = argparse.ArgumentParser()
    else:
        parser = GooeyParser(description="Image Rectification")

    # arguments
    if not gooey:
        parser.add_argument("--input", "-i",
                            action="store",
                            dest="input",
                            default="../../doc/average.png",
                            required=False,
                            help="Input Image.",)

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
                            default="rectified.tiff",
                            help="Rectified image in geotiff format.")

    else:  # add the same thing but a nicer widget
        parser.add_argument("--input", "-i",
                            action="store",
                            dest="input",
                            required=False,
                            help="Input image.",
                            default="../../doc/average.png",
                            widget='FileChooser')

        parser.add_argument("--camera_matrix", "-mtx",
                            action="store",
                            dest="camera_matrix",
                            required=False,
                            default="../../data/flir_tamron_8mm.json",
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
                            default="rectified.tiff",
                            help="Rectified image in geotiff format.",
                            widget='FileChooser')

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

    # read image
    img = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)

    # undistort
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # read coordinates
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
    ximg, yimg = rectify_image(img, H)
    if error:
        print(f"  -- Re-projection error is {round(error, 1)} pixels")

    # image coordinate points
    XY = np.vstack([ximg.flatten(), yimg.flatten()]).T

    # bounding box
    bbox = args.bbox.split(",")
    bbox = np.array([float(bbox[0]), float(bbox[1]),
                     float(bbox[2]), float(bbox[3])])

    # mask points outside the bounding box
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             linewidth=2, edgecolor='r', facecolor='none')

    insiders = rect.contains_points(XY)
    insiders_idx = np.arange(0, len(XY), 1)[insiders]
    outsiders_idx = np.arange(0, len(XY), 1)[~insiders]
    iimg, jimg = np.unravel_index(outsiders_idx, ximg.shape)

    # copy and mask
    ximg_m = ximg.copy()
    yimg_m = yimg.copy()
    img_m = img.copy()
    dst_m = dst.copy()

    ximg_m[iimg, jimg] = np.ma.masked
    yimg_m[iimg, jimg] = np.ma.masked
    img_m[iimg, jimg, :] = np.ma.masked
    dst_m[iimg, jimg, :] = np.ma.masked

    print("\n  -- Interpolating, please wait...")
    # interpolate
    points = XY[insiders_idx, :]

    dx = float(args.dx)
    dy = float(args.dy)
    grid_x, grid_y = np.meshgrid(np.arange(bbox[0], bbox[0] + bbox[2], dx),
                                 np.arange(bbox[1], bbox[1] + bbox[3], dy))

    values = np.vstack([dst_m[:, :, 0].flatten()[insiders_idx],
                        dst_m[:, :, 1].flatten()[insiders_idx],
                        dst_m[:, :, 2].flatten()[insiders_idx]]).T
    rgb = griddata(points,
                   values,
                   (grid_x, grid_y),
                   method=args.interp_method).clip(0, 255)

    # output
    save_as_geotiff(grid_x, grid_y, dx, dy, rgb, args.epsg, args.output)

    # plot
    if args.show:
        plot(grid_x, grid_y, rgb, gcps=xyz)

    print("\nMy work is done!\n")


if __name__ == '__main__':
    main()
