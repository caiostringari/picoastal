# SCRIPT   : calib_ChArUco.py
# POURPOSE : camera calibration using ChArUco boards.
# AUTHOR   : Caio Eadi Stringari

# arguments
import json
import argparse

import cv2

from glob import glob
from natsort import natsorted

import numpy as np

# from time import sleep

import matplotlib.pyplot as plt


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


if __name__ == '__main__':

    print("\nCamera calibration starting, please wait...\n")

    # argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--input", "-i",
                        action="store",
                        dest="input",
                        required=True,
                        help="Input folder with images.",)

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
                        help="Output JSON file.",)

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

    # read images
    images = natsorted(glob(args.input + "/*{}".format(args.format)))
    print("  -- Found {} {} images.".format(len(images), args.format))

    # loop over all images
    all_corners = []
    all_ids = []
    total_images = 0
    for i, image in enumerate(images):

        print("  - processing image {} of {}".format(i+1, len(images)),
              end="\r")

        # read
        frame = cv2.imread(image)

        # covert to grey scale
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                    frame, ref_corners, ref_ids, (0, 255, 0))

                # append
                all_corners.append(ref_corners)
                all_ids.append(ref_ids)

                if total_images > max_images:
                    print("Got all images I needed, breaking the loop.")
                    break

                total_images += 1

        else:
            pass

        if args.show:
            resize = ResizeWithAspectRatio(frame, height=800)
            cv2.imshow("Camera calibration, pres 'q' to quit.", resize)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Destroy any open CV windows
    cv2.destroyAllWindows()

    sys.exit()

    # calibrate the camera

    # termination criteria
    retval, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, imsize, None, None)

    # undistort
    img = cv2.imread(images[0])
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    fig.tight_layout()
    plt.show()

    # print(mtx)

    # except:
    #     cap.release()
    #
    # cap.release()
    # cv2.destroyAllWindows()

    # main()
