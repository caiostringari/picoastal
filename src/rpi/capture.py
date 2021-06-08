"""
Capture frames using the Raspberry Pi HQ camera.

# SCRIPT   : capture.py
# POURPOSE : Capture a sequence of frames record using the
#            raspberry pi HQ camera.
# AUTHOR   : Caio Eadi Stringari
# DATE     : 14/04/2021
# VERSION  : 1.0
"""

# system
import os
import sys
import subprocess
from time import sleep

# files
from glob import glob
from natsort import natsorted

# dates
import datetime

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
    camera.resolution = cfg["capture"]["resolution"]

    # set camera frame rate [Hz]
    camera.framerate = cfg["capture"]["framerate"]

    # exposure mode
    camera.exposure_mode = cfg["exposure"]["mode"]

    if cfg["exposure"]["set_iso"]:
        camera.iso = cfg["exposure"]["iso"]

    return camera


def run_single_camera(cfg):

    # set camera parameters
    camera = set_camera_parameters(cfg)

    # warm-up the camera
    print(" -- warming up the camera (2 seconds) --")
    sleep(2)

    # capture frames from the camera
    start = datetime.datetime.now()
    duration = cfg["capture"]["duration"]  # total number of seconds

    print("\n capturing {} seconds".format(duration))
    print("\n capture started at {} --".format(start))
    fname = os.path.join(cfg["data"]["output"],
                         start.strftime("%Y%m%d_%H%M%S.h264"))
    camera.start_recording(fname,
                           sei=cfg["h264"]["sei"],
                           sps_timing=cfg["h264"]["sps_timing"],
                           quality=cfg["h264"]["quality"])
    camera.wait_recording(duration)
    camera.stop_recording()
    end = datetime.datetime.now()
    print(" capture finished at {} --".format(end))

    if cfg["post_processing"]["extract_frames"]:
        if cfg["post_processing"]["only_last_frame"]:
            print("\n -- Extracting frames -- \n")
            out = os.path.join(cfg["data"]["output"],
                               start.strftime("%Y%m%d_%H%M"))
            extract_frames(fname, out, start, cfg["data"]["format"],
                           only_last=True)
        else:
            print("\n -- Extracting frames -- \n")
            out = os.path.join(cfg["data"]["output"],
                               start.strftime("%Y%m%d_%H%M"))
            extract_frames(fname, out, start, cfg["data"]["format"])


def extract_frames(inp, out, date, ext, only_last=False):
    """
    Extract all frames from the encoded stream.

    :param inp: Input h.264 file.
    :type inp: str
    :param out: Output path.
    :type out: str
    :param date: Capture date.
    :type date: datetime.datetime
    :return: None
    :rtype: None
    """
    if only_last:
        # make sure output path exists
        os.makedirs(out, exist_ok=True)

        print("\n --- Calling FFMPEG (only last frame) ---\n")
        dt = date.strftime("%Y%m%d_%H%M")
        cmd = "ffmpeg -i {} -vf \"select=\'eq(n,1)\'\" -vframes 1 {}/first.bmp > /dev/null 2>&1".format(inp, out)
        subprocess.call(cmd, shell=True)
        print("\n --- FFMPEG finished extracting frames ---\n")
        print("First frame is:")
        print(os.path.join(out, "first.bmp"))
    else:
        # make sure output path exists
        os.makedirs(out, exist_ok=True)

        # call ffmpeg
        print("\n --- Calling FFMPEG ---\n")
        dt = date.strftime("%Y%m%d_%H%M")
        cmd = "ffmpeg -i {} {}/000000-{}_%06d.{} > /dev/null 2>&1".format(
            inp, out, dt, ext)
        subprocess.call(cmd, shell=True)
        print("\n --- FFMPEG finished extracting frames ---\n")

        # list all frames
        files = natsorted(glob(out + "/*.{}".format(ext)))
        print("\nExtracted files are:\n")
        for file in files:
            print(file)

def main():
    """Call the main program."""
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

    # get the date
    today = datetime.datetime.now()

    # check if current hour is in capture hours
    hour = today.hour
    capture_hours = cfg["data"]["hours"]
    if hour in capture_hours:
        print("Sunlight hours. Starting capture cycle.\n")
    else:
        print("Not enough sunlight at {}. Not starting capture "
              "cycle.".format(today))
        sys.exit()

    # create output folder
    os.makedirs(cfg["data"]["output"], exist_ok=True)

    run_single_camera(cfg)


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
