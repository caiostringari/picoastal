# SCRIPT   : calib_ChArUco.py
# POURPOSE : camera calibration using ChArUco boards.
# AUTHOR   : Caio Eadi Stringari

import os

# arguments
import json
import argparse

import cv2

import numpy as np

from time import sleep

import pickle

# PiCamera
from picamera import PiCamera
from picamera.array import PiRGBArray


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def set_camera_parameters(cfg):
    """
    Set camera parameters.

    All values come from the dict generated from the JSON file.

    :param cfg: JSON instance.
    :type cam: dict
    :return: None
    :rtype: None
    """
    # set camera resolution [width x height]
    camera = PiCamera()
    camera.resolution = cfg["capture"]["resolution"]

    # set camera frame rate [Hz]
    camera.framerate = cfg["capture"]["framerate"]

    # exposure mode
    camera.exposure_mode = cfg["exposure"]["mode"]

    if cfg["exposure"]["set_iso"]:
        camera.iso = cfg["exposure"]["iso"]

    return camera


if __name__ == '__main__':

    print("\nCamera calibration starting, please wait...\n")

    # argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--config", "-i",
                        action="store",
                        dest="config",
                        required=True,
                        help="Camera configuration JSON file.",)

    parser.add_argument("--format", "-fmt",
                        action="store",
                        dest="format",
                        default="png",
                        required=False,
                        help="Image format. Default is png.",)

    parser.add_argument("--show", "-show",
                        action="store_true",
                        dest="show",
                        help="Show the calibration process.",)

    # board definition
    parser.add_argument("--squares_x",
                        action="store",
                        dest="squares_x",
                        default=5,
                        required=False,
                        help="Number of squares in the x direction.")

    parser.add_argument("--squares_y",
                        action="store",
                        dest="squares_y",
                        default=7,
                        required=False,
                        help="Number of squares in the y direction.")

    parser.add_argument("--square_length",
                        action="store",
                        dest="square_length",
                        required=False,
                        default=413,
                        help="Square side length (in pixels).")

    parser.add_argument("--marker_length",
                        action="store",
                        dest="marker_length",
                        required=False,
                        default=247,
                        help="Marker side length (in pixels).")

    parser.add_argument("--dictionary_id",
                        action="store",
                        dest="dictionary_id",
                        default="6X6_250",
                        required=False,
                        help="ArUco Dictionary id.")

    parser.add_argument("--max-images", "-N",
                        action="store",
                        dest="max_images",
                        required=False,
                        default=100,
                        help="Maximum number of images to use.",)

    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        required=True,
                        help="Output pickle file.",)

    args = parser.parse_args()

    max_images = int(args.max_images)

    # parse parameters
    squares_x = int(args.squares_x)  # number of squares in X direction
    squares_y = int(args.squares_y)  # number of squares in Y direction
    square_length = int(args.square_length)  # square side length (in pixels)
    marker_length = int(args.marker_length)  # marker side length (in pixels)
    dictionary_id = args.dictionary_id  # dictionary id

    # create board
    dict_id = getattr(cv2.aruco, "DICT_{}".format(dictionary_id))
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    # create the board instance
    board = cv2.aruco.CharucoBoard_create(
        squares_x, squares_y, square_length, marker_length, dictionary)

    # set camera parameters
    inp = args.config
    if os.path.isfile(inp):
        with open(inp, "r") as f:
            cfg = json.load(f)
        print("\nConfiguration file found, continue...")
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))
    camera = set_camera_parameters(cfg)

    # read the data
    rawCapture = PiRGBArray(camera)

    # warm-up the camera
    print("  -- warming up the camera --")
    sleep(2)
    print("  -- starting now --")

    # store data
    all_corners = []
    all_ids = []
    total_images = 0

    # capture frames from the camera
    for frame in camera.capture_continuous(
            rawCapture, format="bgr", use_video_port=True):

        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array

        # covert to grey scale
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            grey, dictionary)
        cv2.aruco.refineDetectedMarkers(
            grey, board, corners, ids, rejectedImgPoints)

        if len(corners) > 0:  # if there is at least one marker detected

            # refine
            retval, ref_corners, ref_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, grey, board)

            if retval > 5:  # calibrateCameraCharuco needs at least 4 corners

                # draw board on image
                im_with_board = cv2.aruco.drawDetectedCornersCharuco(
                    image, ref_corners, ref_ids, (0, 255, 0))

                # append
                all_corners.append(ref_corners)
                all_ids.append(ref_ids)

                if total_images > max_images:
                    print("Got all images I needed, breaking the loop.")
                    break

                total_images += 1

        else:
            pass

        height = cfg["stream"]["resolution"][1]
        resize = ResizeWithAspectRatio(image, height=height)
        cv2.imshow("Camera calibration, pres 'q' to quit.", resize)

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destroy any open CV windows
    cv2.destroyAllWindows()

    # output
    out = {}
    outfile = open(args.output, 'wb')
    out["corners"] = all_corners
    out["ids"] = all_ids
    out["last_frame"] = im_with_board
    pickle.dump(out, outfile)
    outfile.close()
