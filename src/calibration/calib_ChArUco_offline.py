# SCRIPT   : calib_ChArUco_offline.py
# POURPOSE : camera calibration using ChArUco boards. Calibrate either from
#            a series of images or from a series of detections.
# AUTHOR   : Caio Eadi Stringari

# arguments
import json
import argparse

import cv2

from glob import glob
from natsort import natsorted

import numpy as np

from time import sleep

import json
import pickle

import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':

    print("\nCamera calibration starting, please wait...\n")

    # argument parser
    parser = argparse.ArgumentParser()

    # calibration from detected corners
    parser.add_argument("--from_corners",
                        action="store_true",
                        dest="from_corners",
                        help="Use a pickle file to do the calibration.",)

    # input images
    parser.add_argument("--input", "-i",
                        action="store",
                        dest="input",
                        required=True,
                        help="Input folder with images or pickle file.",)

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

    parser.add_argument("--max_images", "-N",
                        action="store",
                        dest="max_images",
                        required=False,
                        default=100,
                        help="Maximum number of images to use.",)

    parser.add_argument("--stream_height",
                        action="store",
                        dest="stream_height",
                        required=False,
                        default=600,
                        help="Height for the opencv stream window image.",)

    parser.add_argument("--stream_width",
                        action="store",
                        dest="stream_width",
                        required=False,
                        default=800,
                        help="Width for the opencv stream window image.",)

    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        required=True,
                        help="Output filename."
                             "If extension is json, write a json file"
                             "If extension is otherwise, write a pickle file",)

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

    # calibrate from detected corners
    if args.from_corners:

        with open(args.input, 'rb') as f:
            x = pickle.load(f)

        all_corners = x["corners"][0:int(args.max_images)]
        all_ids = x["ids"][0:int(args.max_images)]

        frame = x["last_frame"]
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imsize = frame.shape[:2]

    # do the detection
    else:

        # read images
        images = natsorted(glob(args.input + "/*{}".format(args.format)))
        print("  -- Found {} {} images.".format(len(images), args.format))

        # loop over all images
        all_corners = []
        all_ids = []
        total_images = 0
        for i, image in enumerate(images):

            print("  - processing image {} of {}".format(i + 1, len(images)),
                  end="\r")

            # read
            frame = cv2.imread(image)

            # covert to grey scale
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imsize = grey.shape

            # detect
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                grey, dictionary)
            cv2.aruco.refineDetectedMarkers(
                grey, board, corners, ids, rejectedImgPoints)

            if len(corners) > 0:  # if there is at least one marker detected

                # refine
                retval, ref_corners, ref_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, grey, board)

                # calibrateCameraCharuco needs at least 4 corners
                if retval > 5:

                    # draw board on image
                    im_with_board = cv2.aruco.drawDetectedCornersCharuco(
                        frame, ref_corners, ref_ids, (0, 255, 0))

                    # append
                    all_corners.append(ref_corners)
                    all_ids.append(ref_ids)

                    if total_images > max_images:
                        print("\n  --> Found all images I needed. "
                              "Breaking the loop after {} images.".format(
                                  max_images))
                        break

                    total_images += 1

            else:
                pass

            if args.show:
                rsize = (int(args.stream_width), int(args.stream_height))
                resized = cv2.resize(im_with_board, rsize,
                                     interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Camera calibration, pres 'q' to quit.", resized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Destroy any open CV windows
        cv2.destroyAllWindows()


    print("\n - Starting calibrateCameraCharuco(), this will take a while.")

    # calibrate the camera
    retval, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, imsize, None, None)

    # undistort
    h,  w = grey.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # save the output
    out = {}
    outfile = open(args.output, 'wb')
    out["retval"] = retval
    out["camera_matrix"] = mtx
    out["distortion_coefficients"] = dist
    out["rotation_vectors"] = rvecs
    out["translation_vectors"] = tvecs
    out["corners"] = all_corners
    out["ids"] = all_ids
    if args.output.lower().endswith("json"):
        with open(args.output, 'w') as fp:
            json.dump(out, fp, cls=NumpyEncoder)
    else:
        # out["board"] = board
        out["last_frame"] = frame
        with open(args.output, 'wb') as fp:
            pickle.dump(out, fp)

    if args.show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        fig.tight_layout()
        plt.show()

    print("\nMy work is done!\n")
