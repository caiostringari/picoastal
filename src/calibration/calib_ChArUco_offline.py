# SCRIPT   : calib_ChArUco_offline.py
# POURPOSE : camera calibration using ChArUco boards. Calibrate either from
#            a series of images or from a series of detections.
# AUTHOR   : Caio Eadi Stringari

import os
import sys

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

try:
    import gooey
    from gooey import GooeyParser
except ImportError:
    gooey = None


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
        program_name='ChArUco Board Creator',
        default_size=[800, 480],
        navigation="TABBED",
        show_sidebar=True,
        image_dir=image_dir,
        suppress_gooey_flag=True,
    )

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def drawDetectedCornersCharuco(img, corners, ids, rect_size=3,
                               id_font=cv2.FONT_HERSHEY_DUPLEX,
                               id_scale=0.5, id_color=(255, 255, 0),
                               rect_thickness=1):
    """
    Draw rectangles and IDs to the corners
    Parameters
    ----------
    img : numpy.array()
        Two dimensional image matrix. Image can be grayscale image or RGB image
        including 3 layers. Allowed shapes are (x, y, 1) or (x, y, 3).
    corners : numpy.array()
        Checkerboard corners.
    ids : numpy.array()
        Corners' IDs.
    """

    if ids.size > 0:

        # draw rectangels and IDs
        for (corner, id) in zip(corners, ids):
            _corner = np.squeeze(corner)
            corner_x = int(_corner[0])
            corner_y = int(_corner[1])
            id_text = "{}".format(str(id[0]))
            id_coord = (corner_x + 2 * rect_size, corner_y + 2 * rect_size)
            cv2.rectangle(img, (corner_x - rect_size, corner_y - rect_size),
                          (corner_x + rect_size, corner_y + rect_size),
                          id_color, thickness=rect_thickness)
            cv2.putText(img, id_text, id_coord, id_font, id_scale, id_color)

    return img


@gui_decorator
def main():
    print("\nCamera calibration starting, please wait...\n")

    # Argument parser
    if not gooey:
        parser = argparse.ArgumentParser()
    else:
        parser = GooeyParser(description="Offline Camera Calibration")

    # input configuration file
    if not gooey:
        parser.add_argument("--input", "-i",
                            action="store",
                            dest="input",
                            required=True,
                            help="Input folder with images "
                                 "or pickle file with corners and ids.",)
    else:
        parser.add_argument("--input", "-i",
                            action="store",
                            dest="input",
                            required=True,
                            help="Input folder with images "
                                 "or pickle file with corners and ids.",
                            widget='DirChooser')

    # calibration from detected corners
    parser.add_argument("--from_corners",
                        action="store_true",
                        dest="from_corners",
                        help="Use a pickle file with corners and ids to do the"
                             " calibration.",)

    # input images format
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
                        default=25,
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
        images = natsorted(glob(args.input + "/*.{}".format(args.format)))
        print("  -- Found {} {} images.".format(len(images), args.format))

        if len(images) > max_images:
            print("   -- Enough images. Starting now.")
        else:
            sys.exit(
                "   -- Not enough images or trying to calibrate from a file.")

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
                        frame, ref_corners, ref_ids, (0, 0, 255))
                    im_with_board = cv2.aruco.drawDetectedMarkers(
                        im_with_board, corners, ids)

                    # save 
                    cv2.imwrite(args.output + "/" + str(total_images) + ".png",
                              im_with_board)

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
                try:
                    resized = cv2.resize(im_with_board, rsize,
                                         interpolation=cv2.INTER_LINEAR)
                except:
                    resized = cv2.resize(frame, rsize,
                                         interpolation=cv2.INTER_LINEAR)
                    
                cv2.imshow("Camera calibration, pres 'q' to quit.", resized)
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break

        # Destroy any open CV windows
        cv2.destroyAllWindows()

    # sys.exit()
    print("\n - Starting calibrateCameraCharuco(), this will take a while.")

    # calibrate the camera
    retval, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, imsize, None, None)

    print(f"\n    - Calibration error: {round(retval, 2)} units")

    # undistort
    h,  w = grey.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # save the output
    out = {}
    outfile = open(args.output, 'wb')
    out["error"] = retval
    out["camera_matrix"] = mtx
    out["distortion_coefficients"] = dist
    out["rotation_vectors"] = rvecs
    out["translation_vectors"] = tvecs
    out["corners"] = all_corners
    out["ids"] = all_ids
    out["chessboard_size"] = board.getChessboardSize()
    out["marker_length"] = board.getMarkerLength()
    out["square_length"] = board.getSquareLength()
    
    if args.output.lower().endswith("json"):
        with open(args.output, 'w') as fp:
            json.dump(out, fp, cls=NumpyEncoder)
    else:
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


if __name__ == '__main__':

    main()
