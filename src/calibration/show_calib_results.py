# SCRIPT   : show_calib_results.py
# POURPOSE : Display the results of a calibration
# AUTHOR   : Caio Eadi Stringari

import argparse

import cv2

import pickle

import matplotlib.pyplot as plt

import sys


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()

    # board definition
    parser.add_argument("--input", "-i",
                        action="store",
                        dest="input",
                        required=True,
                        help="Input file. Only pickle files are supported.")

    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        required=True,
                        help="Figure file name.")

    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        x = pickle.load(f)

    keys = ["last_frame", "distortion_coefficients", "camera_matrix",
            "translation_vectors", "rotation_vectors", "corners", "ids",
            ]

    for key in keys:
        if key not in x.keys():
            print("Fatal: Key {} not present in input data.".format(key))
            sys.exit()

    # load calibration data
    mtx = x["camera_matrix"]
    dist = x["distortion_coefficients"]
    rvecs = x["rotation_vectors"]
    tvecs = x["translation_vectors"]
    frame = x["last_frame"]
    size = x["board_size"]

    # undistort
    h,  w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    #
    # try:
    #     corners = x["corners"]
    #     compute_errors = True
    # except Exception:
    #     compute_errors = False
    #     print("warning: cannot compute errors")
    #     pass
    #
    # if compute_errors:
    #     mean_error = 0
    #     for i in range(len(corners)):
    #         imgpoints2, _ = cv2.projectPoints(corners[i], rvecs[i], tvecs[i], mtx, dist)
    #         error = cv2.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    #         mean_error += error

    im_with_board = cv2.aruco.drawDetectedCornersCharuco(
        frame, x["corners"][-1], x["ids"][-1], (0, 255, 0))

    im_with_board = cv2.drawChessboardCorners(
        im_with_board, (7, 5),
        x["corners"][-1], False)

    # im_with_board = cv2.aruco.drawDetectedMarkers(frame, x["corners"][-1], x["ids"][-1], (0, 255, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    ax1.imshow(im_with_board)
    ax2.imshow(dst)

    ax1.set_title("Last Frame")
    ax2.set_title("Undistorted Last Frame")

    fig.tight_layout()

    plt.show()




    # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # imsize = frame.shape[:2]
