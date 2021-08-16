"""
Find people in a given image.

Based on https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi

# SCRIPT   : people_detector_offline.py
# POURPOSE : Find people in a given image.
# AUTHOR   : Caio Eadi Stringari
# DATE     : 02/07/2021
# VERSION  : 1.0
"""

import os

import argparse

import re

import cv2

import numpy as np

import pandas as pd

from glob import glob
from natsort import natsorted

from tflite_runtime.interpreter import Interpreter

from tqdm import tqdm


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []

    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }

            results.append(result)

    return results


def get_bbox_and_label(results, model_labels, image_width, image_height):
    """Draws the bounding box and label for each object in the results."""

    bboxes = []
    labels = []
    scores = []
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * image_width)
        xmax = int(xmax * image_width)
        ymin = int(ymin * image_height)
        ymax = int(ymax * image_height)

        # Overlay the box, label, and score on the camera preview
        bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
        label = model_labels[obj['class_id']]
        score = obj['score']

        bboxes.append(bbox)
        labels.append(label)
        scores.append(score)

    return bboxes, labels, scores


def main():
    "Call the main program."

    parser = argparse.ArgumentParser(
        description='Find people in a given series of images')

    parser.add_argument('--model', "-M",
                        dest='model',
                        help='pre-trained model in .tflite format',
                        required=True,
                        action='store')

    parser.add_argument('--model_labels', "-ML",
                        dest='model_labels',
                        help='List of clases in .csv format',
                        required=True,
                        action='store')

    parser.add_argument("--input", "-i", "--frames", "-frames",
                        action="store",
                        dest="input",
                        required=True,
                        help="Input path with frames.",)

    parser.add_argument("--image_format", "-if", "-image_format",
                        action="store",
                        dest="image_format",
                        required=False,
                        default="jpg",
                        help="Image format. Default is jpg.",)

    parser.add_argument("--region_of_interest", "-roi",
                        nargs=4,
                        action="store",
                        dest="roi",
                        required=False,
                        default=[None, None, None, None],
                        help="Region of interest. Default is none."
                        "Format is top_left dx, dy.",)

    parser.add_argument("--threshold", "-threshold", "--trx", "-trx",
                        action="store",
                        dest="threshold",
                        default=0.3,
                        required=False,
                        help="Threshold for detection. Default is 0.3",)

    parser.add_argument("--output", "-output", "-o",
                        action="store",
                        dest="output",
                        required=True,
                        help="Output csv file with bounding boxes.",)

    parser.add_argument("--display",
                        action="store_true",
                        dest="show",
                        help="Show evolution on screen.",)

    parser.add_argument("--display_scale", "-scale",
                        action="store",
                        dest="show_scale",
                        default=2,
                        required=False,
                        help="Scale factor to reduce images on screen.",)

    parser.add_argument("--save", "--save_frames",
                        action="store_true",
                        dest="save",
                        help="Save frames with detections.",)

    parser.add_argument("--save_path",
                        action="store",
                        dest="save_path",
                        default="detections",
                        required=False,
                        help="Where to save frames with detections.",)

    args = parser.parse_args()

    model_labels = args.model_labels
    model = args.model
    threshold = float(args.threshold)
    data = args.input
    image_format = args.image_format
    show = args.show
    scale = int(args.show_scale)
    output = args.output
    save_frames = args.save
    out_frame_path = args.save_path

    if save_frames:
        os.makedirs(out_frame_path, exist_ok=True)  # make this folder exists

    # initiate the model
    model_labels = load_labels(model_labels)
    interpreter = Interpreter(model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']

    # get images
    images = natsorted(glob(data + f"/*.{image_format}"))

    camera_width, camera_height = (cv2.imread(images[0]).shape[1],
                                   cv2.imread(images[0]).shape[0])

    # define region of interest. Format is top_left dx, dy.
    roi = args.roi
    if not roi[0]:
        roi = [0, 0, camera_height, camera_width]
    else:
        roi = np.array(roi).astype(int)
        img = cv2.imread(images[0])
        img = img[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3], :]
        camera_width, camera_height = (img.shape[1], img.shape[0])

    out_bboxes = []
    out_labels = []
    out_scores = []
    out_frame = []

    # start progess bar,
    if not show:
        pbar = tqdm(total=len(images))

    for i, image in enumerate(images):

        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

        # cut to ROI
        img = img[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3], :]

        img_for_model = cv2.resize(img, (input_width, input_height),
                                   interpolation=cv2.INTER_LINEAR)

        results = detect_objects(interpreter, img_for_model, threshold)

        # get boxes
        bboxes, labels, scores = get_bbox_and_label(results,
                                                    model_labels,
                                                    camera_width,
                                                    camera_height)

        # draw bounding boxes
        annotated = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if labels:
            for j in range(len(labels)):
                if labels[j] == "person":
                    x, y, dx, dy = bboxes[j]
                    annotated = cv2.rectangle(annotated, (x, y),
                        (x + dx, y + dy), (0, 255, 0), 2)

                    # append to output
                    out_bboxes.append([x, y, dx, dy])
                    out_labels.append(labels[j])
                    out_scores.append(scores[j])
                    out_frame.append(i)

        if save_frames:
            cv2.imwrite("{}/detection_{}.{}".format(
                out_frame_path, str(i).zfill(6), image_format), annotated)

        if show:
            wname = "People detector, pres 'q' to quit."
            # resize and show
            size = (camera_width // scale, camera_height // scale)
            dst = cv2.resize(annotated, size, interpolation=cv2.INTER_LINEAR)
            cv2.imshow(wname, dst)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            pbar.update()

    # output
    if out_bboxes:
        out_bboxes = np.array(out_bboxes)
        cols = ["top_left_x", "top_left_y", "dx", "dy"]
        df = pd.DataFrame(out_bboxes, columns=cols)
        df["label"] = out_labels
        df["score"] = out_scores
        df["frame"] = out_frame
    else:
        df = pd.DataFrame()  # no data

    df.to_csv(output)

    # destroy any open CV windows
    cv2.destroyAllWindows()

    # close pbar
    if not show:
        pbar.close()

    print("\nMy work is done!\n")


if __name__ == '__main__':
    main()
