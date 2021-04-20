"""
# SCRIPT   : stream.py
# POURPOSE : Stream the camera to the screen.
# AUTHOR   : Caio Eadi Stringari
# DATE     : 14/04/2021
# VERSION  : 1.0
"""

# system
import os
from time import sleep

# arguments
import json
import argparse

# PiCamera
from picamera import PiCamera
from picamera.array import PiRGBArray

# OpenCV
import cv2


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
    camera.resolution = cfg["stream"]["resolution"]
    
    # set camera frame rate [Hz]
    camera.framerate = cfg["stream"]["framerate"]

    # exposure mode
    camera.exposure_mode = cfg["exposure"]["mode"]
    
    if cfg["exposure"]["set_iso"]:
        camera.iso = cfg["exposure"]["iso"]

    return camera
    

def run_single_camera(cfg):
    """
    Capture frames and display them on the screen
    """

    # set camera parameters
    camera = set_camera_parameters(cfg)

    # read the data
    rawCapture = PiRGBArray(camera)

    # warm-up the camera
    print("  -- warming up the camera --")
    sleep(2)
    print("  -- starting now --")

    # capture frames from the camera
    for frame in camera.capture_continuous(
            rawCapture, format="bgr", use_video_port=True):
        
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        
        # show the frame
        cv2.imshow("Camera stream - press 'q' to quit.", image)
        
        key = cv2.waitKey(1) & 0xFF
        
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break


def main():
    # verify if the configuraton file exists
    # if it does, then read it
    # else, stop
    inp = args.config[0]
    if os.path.isfile(inp):
        with open(inp, "r") as f:
            cfg = json.load(f)
        print("\nConfiguration file found, continue...")
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))

    # start the stream
    print("\nStreaming the camera")
    
    run_single_camera(cfg)

    print("Stream has ended.")


if __name__ == "__main__":


    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--configuration-file", "-cfg", "-i",
                        nargs=1,
                        action="store",
                        dest="config",
                        required=True,
                        help="Configuration JSON file.",)

    args = parser.parse_args()

    # call the main program
    main()