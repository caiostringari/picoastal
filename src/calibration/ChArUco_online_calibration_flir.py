# coding=utf-8
# =============================================================================
# Copyright (c) 2001-2019 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
#
# Acquisition.py shows how to acquire images. It relies on
# information provided in the Enumeration example. Also, check out the
# ExceptionHandling and NodeMapInfo examples if you haven"t already.
# ExceptionHandling shows the handling of standard and Spinnaker exceptions
# while NodeMapInfo explores retrieving information from various node types.
#
# This example touches on the preparation and cleanup of a camera just before
# and just after the acquisition of images. Image retrieval and conversion,
# grabbing image data, and saving images are all covered as well.
#
# Once comfortable with Acquisition, we suggest checking out
# AcquisitionMultipleCamera, NodeMapCallback, or SaveToAvi.
# AcquisitionMultipleCamera demonstrates simultaneously acquiring images from
# a number of cameras, NodeMapCallback serves as a good introduction to
# programming with callbacks and events, and SaveToAvi exhibits video creation.

# This example has been modified by Caio Stringari for the PiCoastal Project.

# This scrip only streams camera data to the screen.

# - Fixed all PEP8 issues.
# - Add a JSON config file that has options to set frame rate,
#   capture interval, image ROI, dx, dy and more.

import os
import sys
import time

# dates
import datetime

# arguments
import json
import argparse

import cv2

import pickle

# PySpin
import PySpin

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def set_camera_parameters(cam, nodemap, nodemap_tldevice, fps=5, height=1080,
                          width=1920, offsetx=80, offsety=236):
    """
    Set capture parameters.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :param fps: Frames per second.
    :param height: Image height.
    :param width: Image width.
    :param offsetx: Image offset x.
    :param offsety: Image offset y.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :type fps: int Default = 5.
    :type height: int Default = 1080.
    :type width: int Default = 1920.
    :type offsetx: int Default = 80.
    :type offsety: int. Default = 236.
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print("\n*** SETTING CAMERA PARAMETERS ***\n")
    try:
        result = True

        # *** Frame rate ***

        # allow to set frqme rate
        node_acquisition_fps_auto = PySpin.CEnumerationPtr(
                    nodemap.GetNode("AcquisitionFrameRateAuto"))
        node_acquisition_fps_auto.SetIntValue(0)

        # get the current frame rate
        i = cam.AcquisitionFrameRate.GetValue()
        print("Current frame rate: %d " % i)
        # set the new frame rate
        cam.AcquisitionFrameRate.SetValue(int(fps))
        i = cam.AcquisitionFrameRate()
        print("Frame rate set to: %d " % i)

        # *** Image size ***

        # get the current image height
        i = cam.Height.GetValue()
        print("Current image height: %d " % i)
        # get the current image height
        cam.Height.SetValue(height)
        i = cam.Height.GetValue()
        print("Image height set to: %d " % i)

        # get the current image width
        i = cam.Width.GetValue()
        print("Current image width: %d " % i)
        # get the current image width
        cam.Width.SetValue(width)
        i = cam.Width.GetValue()
        print("Image width set to: %d " % i)

        # *** Image offset ***

        # get the current image offsetx
        i = cam.OffsetX.GetValue()
        print("Current OffsetX : %d " % i)
        # set the current image offsetX
        cam.OffsetX.SetValue(offsetx)
        i = cam.OffsetX.GetValue()
        print("Image OffsetX set to: %d " % i)

        # get the current image offsetY
        i = cam.OffsetY.GetValue()
        print("Current OffsetX : %d " % i)
        # set the current image offsetY
        cam.OffsetY.SetValue(offsety)
        i = cam.OffsetY.GetValue()
        print("Image OffsetX set to: %d " % i)

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        return False

    return result


def acquire_images(cam, nodemap, nodemap_tldevice):
    """
    Acquires and saves N images from a device.

    :param cam: Camera to acquire images from.
    :param nodemap: Device nodemap.
    :param nodemap_tldevice: Transport layer device nodemap.
    :type cam: CameraPtr
    :type nodemap: INodeMap
    :type nodemap_tldevice: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print("\n*** IMAGE ACQUISITION ***\n")
    try:
        result = True

        # Set acquisition mode to continuous
        #
        #  *** NOTES ***
        # Because the example acquires and saves 10 images, setting
        # acquisition mode to continuous lets the example finish. If set to
        # single frame or multiframe (at a lower number of images),
        # the example would just hang. This would happen because the example
        # has been written to acquire N images while the camera would have
        # been programmed to retrieve less than that.
        #
        # Setting the value of an enumeration node is slightly more complicated
        # than other node types. Two nodes must be retrieved: first, the
        # enumeration node is retrieved from the nodemap; and second, the entry
        # node is retrieved from the enumeration node. The integer value of the
        # entry node is then set as the new value of the enumeration node.
        #
        # Notice that both the enumeration and the entry nodes are checked for
        # availability and readability/writability. Enumeration nodes are
        # generally readable and writable whereas their entry nodes are only
        # ever readable.
        #
        #  Retrieve enumeration node from nodemap

        sNodemap = cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        # In order to access the node entries, they have to be casted to a
        # pointer type (CEnumerationPtr here)
        node_acquisition_mode = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionMode"))
        if not PySpin.IsAvailable(node_acquisition_mode) \
                or not PySpin.IsWritable(node_acquisition_mode):
            print(
                "Unable to set acquisition mode to continuous"
                "(enum retrieval)."
                "Aborting...")
            return False

        # Retrieve entry node from enumeration node
        node_acquisition_mode_continuous =  \
            node_acquisition_mode.GetEntryByName("Continuous")
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) \
                or not PySpin.IsReadable(node_acquisition_mode_continuous):
            print(
                "Unable to set acquisition mode to continuous"
                "(entry retrieval). Aborting...")
            return False

        # Retrieve integer value from entry node
        acquisition_mode_continuous =  \
            node_acquisition_mode_continuous.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        print("Acquisition mode set to continuous...")

        # Begin acquiring images
        #
        # *** NOTES ***
        # What happens when the camera begins acquiring images depends on the
        # acquisition mode. Single frame captures only a single image, multi
        # frame catures a set number of images, and continuous captures a
        # continuous stream of images. Because the example calls for the
        # retrieval of 10 images, continuous mode has been set.
        #
        # *** LATER ***
        # Image acquisition must be ended when no more images are needed.
        cam.BeginAcquisition()

        print("\n *** Acquiring images ***\n")

        # Retrieve device serial number for filename
        #
        # *** NOTES ***
        # The device serial number is retrieved in order to keep cameras from
        # overwriting one another. Grabbing image IDs could also accomplish
        # this.
        device_serial_number = ""
        node_device_serial_number = PySpin.CStringPtr(
            nodemap_tldevice.GetNode("DeviceSerialNumber"))
        if PySpin.IsAvailable(node_device_serial_number) and \
                PySpin.IsReadable(node_device_serial_number):
            device_serial_number = node_device_serial_number.GetValue()
            # print("\nDevice serial number retrieved as %s...\n" %
            #  device_serial_number)

        # Retrieve, and display

        # store data
        all_corners = []
        all_ids = []
        total_images = 0

        while (True):
            try:

                # Retrieve next received image
                #
                # *** NOTES ***
                # Capturing an image houses images on the camera buffer.
                # Trying to capture an image that does not exist will hang the
                # camera.
                #
                # *** LATER ***
                # Once an image from the buffer is saved and/or no longer
                # needed, the image must be released in order to keep the
                # buffer from filling up.
                image_result = cam.GetNextImage()

                # Ensure image completion
                #
                # *** NOTES ***
                # Images can easily be checked for completion. This should be
                # done whenever a complete image is expected or required.
                # Further, check image status for a little more insight into
                # why an image is incomplete.
                if image_result.IsIncomplete():
                    print("Image incomplete with image status %d ..." %
                          image_result.GetImageStatus())

                else:

                    # Print image information; height and width recorded in
                    # pixels
                    #
                    # *** NOTES ***
                    # Images have quite a bit of available metadata including
                    # things such as CRC, image status, and offset values, to
                    # name a few.
                    width = image_result.GetWidth()
                    height = image_result.GetHeight()
                    # print("Grabbed Image %d, width = %d, height = %d" %
                    #       (i, width, height))

                    # Convert image to RGB8
                    #
                    # *** NOTES ***
                    # Images can be converted between pixel formats by using
                    # the appropriate enumeration value. Unlike the original
                    # image, the converted one does not need to be released as
                    # it does not affect the camera buffer.
                    #
                    # When converting images, color processing algorithm is an
                    # optional parameter.
                    image_converted = image_result.Convert(
                        PySpin.PixelFormat_RGB8, PySpin.HQ_LINEAR)
                    image_data = image_converted.GetNDArray()
                    image_data = image_data.reshape(height, width, 3)

                    # search for the ChArUco board

                    # covert to grey scale
                    grey = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

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
                                image_data, ref_corners, ref_ids, (0, 255, 0))

                            # does not work very well
                            # im_with_board = cv2.drawChessboardCorners(
                            #     im_with_board, board.getChessboardSize(),
                            #     ref_corners, True)

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

                    # image to be displayed
                    try:
                        stream_img = im_with_board
                    except Exception:
                        stream_img = image_data  # image without the board.

                    # Display the resulting frame
                    resized = cv2.resize(stream_img, (stream_width,
                                                      stream_height))
                    cv2.imshow("Camera stream - press 'q' to quit.",
                               cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                    #  Release image
                    #
                    #  *** NOTES ***
                    # Images retrieved directly from the camera  (i.e.
                    # non-converted images) need to be released in order
                    # to keep from filling the buffer.
                    image_result.Release()
                    # print("")

            except PySpin.SpinnakerException as ex:
                print("Error: %s" % ex)
                return False

        # End acquisition
        #
        # *** NOTES ***
        # Ending acquisition appropriately helps ensure that devices clean up
        # properly and do not need to be power-cycled to maintain integrity.
        cam.EndAcquisition()
        cv2.destroyAllWindows()

        if args.calibrate_on_device:

            print(
                "\n - Starting calibrateCameraCharuco(), this will take a while.")

            # calibrate the camera
            imsize = grey.shape
            retval, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, board, imsize, None, None)

            # output the results of the calibration
            out = {}
            outfile = open(args.output, 'wb')
            out["retval"] = retval
            out["camera_matrix"] = mtx
            out["distortion_coefficients"] = dist
            out["rotation_vectors"] = rvecs
            out["translation_vectors"] = tvecs
            out["corners"] = all_corners
            out["ids"] = all_ids
            # out["board"] = board

            if args.output.lower().endswith("json"):
                with open(args.output, 'w') as fp:
                    json.dump(out, fp, cls=NumpyEncoder)
            else:
                out["last_frame"] = im_with_board
                with open(args.output, 'wb') as fp:
                    pickle.dump(out, fp)

            # display the results

            # undistort
            h,  w = stream_img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h))

            # print("\n")
            # print(mtx)
            # print("\n")

            dst = cv2.undistort(stream_img, mtx, dist, None, newcameramtx)
            rsize = (stream_width, stream_height)
            resized = cv2.resize(dst, rsize,
                                 interpolation=cv2.INTER_LINEAR)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            cv2.imshow("Undistorted image. Displaying for 20 seconds.",
                       resized)
            cv2.waitKey(20000)
            cv2.destroyAllWindows()

        # output the corners and ids.
        else:
            out = {}
            outfile = open(args.output, 'wb')
            out["corners"] = all_corners
            out["ids"] = all_ids
            out["last_frame"] = im_with_board
            pickle.dump(out, outfile)
            outfile.close()

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        return False

    return result


def print_device_info(nodemap):
    """
    Print the device information of the camera from the transport layer.

    Please see NodeMapInfo example for more in-depth
    comments on printing device information from the nodemap.

    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """
    print("\n*** DEVICE INFORMATION ***\n")

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(
            nodemap.GetNode("DeviceInformation"))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(
                node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print("%s: %s" % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(
                                  node_feature) else "Node not readable"))

        else:
            print("Device control information not available.")

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        return False

    return result


def run_single_camera(cam, cfg):
    """
    Run the camera.

    This function acts as the body of the example; please see NodeMapInfo
    example for more in-depth comments on setting up cameras.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Retrieve TL device nodemap and print device information
        nodemap_tldevice = cam.GetTLDeviceNodeMap()

        result &= print_device_info(nodemap_tldevice)

        # Initialize camera
        cam.Init()

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        # Set camera parameters
        result &= set_camera_parameters(cam, nodemap, nodemap_tldevice,
                                        fps=cfg["stream"]["framerate"],
                                        height=cfg["capture"]["resolution"][1],
                                        width=cfg["capture"]["resolution"][0],
                                        offsetx=cfg["capture"]["offset"][0],
                                        offsety=cfg["capture"]["offset"][1])

        # set stream mode to auto
        # ?
        # node_cmd(serial, 'TLStream.StreamBufferCountMode', 'SetValue', 'RW', 'PySpin.StreamBufferCountMode_Manual')


        # print("--- sleeping for 5 seconds ---")
        # time.sleep(5)
        # Acquire images
        result &= acquire_images(cam, nodemap, nodemap_tldevice)

        # Deinitialize camera
        cam.DeInit()

    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
        result = False

    return result


def main():
    """
    Run the main program.

    Example entry point; please see Enumeration example for more in-depth
    comments on preparing and cleaning up the system.

    :return: True if successful, False otherwise.
    :rtype: bool
    """
    # verify if the input file exists,
    # if it does, then read it
    inp = args.config[0]
    if os.path.isfile(inp):
        with open(inp, "r") as f:
            cfg = json.load(f)
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))

    # frame rate
    fps = cfg["stream"]["framerate"]

    # window size - Note thei are global variables
    size = cfg["stream"]["resolution"]
    global stream_width
    global stream_height
    stream_width = int(size[0])
    stream_height = int(size[1])

    size = cfg["capture"]["resolution"]
    global image_width
    global image_height
    image_width = int(size[0])
    image_height = int(size[1])

    result = True

    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print("Library version: %d.%d.%d.%d" %
          (version.major, version.minor, version.type, version.build))

    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()

    num_cameras = cam_list.GetSize()

    print("\nNumber of cameras detected: %d" % num_cameras)

    # Finish if there are no cameras
    if num_cameras == 0:

        # Clear camera list before releasing system
        cam_list.Clear()

        # Release system instance
        system.ReleaseInstance()

        print("Not enough cameras!")
        input("Done! Press Enter to exit...")
        return False

    # Only support for 1 camera at the moment
    elif num_cameras == 1:
        pass

    else:
        raise IOError("Only support one camera at the moment")
        return False

    # Run example on each camera
    for i, cam in enumerate(cam_list):

        print("\nRunning capture cycle for camera %d..." % i)

        result &= run_single_camera(cam, cfg)
        print("\nCamera %d example complete... \n" % i)
        print("My work is done!")

    # Release reference to camera
    # NOTE: Unlike the C++ examples, we cannot rely on pointer objects
    # being automatically
    # cleaned up when going out of scope.
    # The usage of del is preferred to assigning the variable to None.
    del cam

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    # input("Done! Press Enter to exit...")
    return result


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

    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        required=True,
                        help="Output pickle file.",)

    parser.add_argument("--calibrate_on_device",
                        action="store_true",
                        dest="calibrate_on_device",
                        help="Will calibrate on device, if parsed.",)

    args = parser.parse_args()

    # call the main program

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

    main()
