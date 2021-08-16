"""
Active wave breaking segmentation using the neuralnets from
https://github.com/caiostringari/deepwaves

Warning: These models were trained on offshore data. We are applying them to
suf zone data. Do not expect greate results.

PROGRAM   : offline_wave_breaking_segmentation.py
POURPOSE  : Segmente active wave breaking
AUTHOR    : Caio Eadi Stringari
EMAIL     : caio.stringari@gmail.com
v1.0      : 05/07/2021 [Caio Stringari]
"""

import os
import argparse

import math

from copy import copy

from glob import glob
from natsort import natsorted

import numpy as np

from skimage.util import view_as_blocks

import pandas as pd

# import tensorflow as tf
from tflite_runtime.interpreter import Interpreter

# progress bar
from tqdm import tqdm

# plot
# import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams['patch.edgecolor'] = "k"

# tf.get_logger().setLevel('INFO')


def ensure_shape(img, block_shape):
    """
    Ensure that image shape is compatible with view_as_blocks.

    Block shape must be a power of 2 and will be coerced if not.
    Image will  be coerced to a shape that divides into block shape evenly.

    Parameters:
    ----------
    img : np.ndarray
        Input image.
    block_shape : list-like
        Block shape.

    Returns:
    -------
    img : np.ndarray
        Output image
    block_shape : list-like
        New block shape.
    """
    block_shape = np.array(block_shape)
    if not np.log2(block_shape[0]).is_integer():
        block_shape[0] = closest_power2(block_shape[0])
        print("     warning: block shape has been updated to a power of 2.")
    if not np.log2(block_shape[1]).is_integer():
        block_shape[1] = closest_power2(block_shape[1])
        print("     warning: block shape has been updated to a power of 2.")

    newsize = (closest_power2(img.shape[0]), closest_power2(img.shape[1]))
    img = img[0:newsize[0], 0:newsize[1], :]

    return img, block_shape


def closest_power2(x):
    """Get the closest power of 2 checking if the 2nd binary number is a 1."""
    op = math.floor if bin(x)[3] != "1" else math.ceil
    return 2**(op(math.log(x, 2)))


def display_mask(val_preds, i):
    """Display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def make_plot(img, df, roi_patch=False, total_frames=-1, out_path="plt",
              block_shape=[256, 256]):
    """Plot the results."""

    # plot
    fig, ax = plt.subplots(figsize=(img.shape[0]//100, img.shape[1]//100))
    ax.imshow(img)
    ax.scatter(df["j"], df["i"], marker=".", s=1, color="r", linewidths=1,
               alpha=0.1, rasterized=True)

    # region of interest
    if roi_patch:
        ax.add_patch(copy(roi_patch))

    # search blocks
    x0 = roi_patch.get_bbox().x0
    y0 = roi_patch.get_bbox().y0
    w = roi_patch.get_bbox().width
    h = roi_patch.get_bbox().height

    x = np.arange(x0, x0 + w + block_shape[0], block_shape[0])
    y = np.arange(y0, y0 + h + block_shape[1], block_shape[1])
    x, y = np.meshgrid(x, y)
    ax.scatter(x, y, marker="+", color="w", zorder=50, s=20, linewidths=1)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)

    ax.set_xlabel(r"$i$ [pixel]")
    ax.set_ylabel(r"$j$ [pixel]")
    ax.set_aspect("equal")

    txt = "Frame {} of {}".format(str(df["frame"].values[0] + 1).zfill(5),
                                  str(total_frames).zfill(5))
    ax.text(0.02, 0.98, txt, color="white",
            va="top", zorder=100, transform=ax.transAxes,
            ha="left", fontsize=10,
            bbox=dict(boxstyle="square", ec="none", fc="0.1",
                      lw=0, alpha=0.7))
    fig.tight_layout()

    # save
    fname = str(df["frame"].values[0]).zfill(6) + ".png"
    plt.savefig(os.path.join(out_path, fname), dpi=150,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()


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


def main():
    """Call the main program."""
    # i/o
    model = args.model[0]  # pre-trained model
    frames = args.input[0]  # frames to be segmented
    output = args.output[0]  # output csv file

    # plots
    save_plots = args.save_plots
    plot_path = args.plot_path[0]

    # create output
    if save_plots:
        os.makedirs(plot_path, exist_ok=True)

    # load the model
    interpreter = Interpreter(model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']
    # M = tf.keras.models.load_model(model)

    # --- parameters ---
    roi = np.array(args.region_of_interest).astype(int)

    # get the input image size
    size = (input_height, input_width)

    # verify if the input path exists,
    # if it does, then get the frame names
    if os.path.isdir(frames):
        frames = natsorted(glob(frames + "/*"))
    else:
        raise IOError("No such file or directory \"{}\"".format(frames))

    # select from which frame to start processing and how
    # many frames to process
    start = int(args.start[0])
    if int(args.nframes[0]) == -1:
        N = len(frames)
    else:
        N = int(args.nframes[0])
    total_frames = len(frames)
    frames = frames[start:start + N]

    # --- define region of interest ---
    roi_patch = patches.Rectangle((roi[0], roi[1]),
                                  roi[2], roi[3],
                                  linewidth=1,
                                  edgecolor="lawngreen",
                                  facecolor="none",
                                  linestyle="-",
                                  zorder=20)

    # --- loop over frames ---

    pbar = tqdm(total=len(frames))

    DF = []  # store ALL the data
    for k, frame in enumerate(frames):

        # load image
        img = plt.imread(frame)

        # create a mask
        mask = np.zeros(img.shape).astype(int)
        mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1

        # mask -- this will be the working image
        imk = img * mask

        # compute
        imk, block_shape = ensure_shape(imk, (size[0], size[1]))
        view = view_as_blocks(imk, (size[0], size[1], 3))

        # loop over image blocks
        dfs = []
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):

                # target block
                blk = view[i, j, :, :, :]

                # update indexes to keep track of the pixels
                # in the original image
                i1 = block_shape[0] * i
                # i2 = i1 + block_shape[0]
                j1 = block_shape[1] * j
                # j2 = j1 + block_shape[1]

                # if NOT all black, apply the model
                if not np.all(blk == 0):

                    # predict
                    # very important to normalize your data !
                    set_input_tensor(interpreter, blk / 255)
                    interpreter.invoke()
                    pred = get_output_tensor(interpreter, 0)
                    prd = np.squeeze(np.argmax(pred, axis=-1))

                    # get only white pixels
                    ipx, jpx = np.where(prd)  # gets where prd == 1

                    df = pd.DataFrame(
                        np.vstack([ipx, jpx]).T, columns=["i", "j"])
                    df["i"] = df["i"] + i1
                    df["j"] = df["j"] + j1

                    dfs.append(df)

        # merge all blocks
        df = pd.concat(dfs)
        df["frame"] = k
        # print(df)

        # save plots if asked
        if save_plots:
            try:
                make_plot(img, df, block_shape=block_shape, out_path=plot_path,
                          total_frames=total_frames, roi_patch=roi_patch)
            except Exception:
                pbar.write(f"warning: could not process frame {k}")

        # append to output
        DF.append(df[["i", "j", "frame"]])

        pbar.update()

    # merge everything
    DF = pd.concat(DF)
    DF.to_csv(output, chunksize=2**12, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Predict active wave breaking segmentation')

    parser.add_argument('--model', "-M",
                        nargs=1,
                        dest='model',
                        help='pre-trained model in .h5 format',
                        required=True,
                        action='store')

    parser.add_argument("--input", "-i", "--frames", "-frames",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input path with frames.",)

    parser.add_argument("--regex", "-re",
                        nargs=1,
                        action="store",
                        dest="regex",
                        required=False,
                        default=["[0-9]{6,}"],
                        help="Regex to used when looking for files.",)

    parser.add_argument("--region-of-interest", "-roi", "-R",
                        nargs=4,
                        action="store",
                        dest="region_of_interest",
                        required=False,
                        default=[256, 256, 1024, 512],
                        help="Region of Interest. Format is top_left dx, dy.",)

    parser.add_argument("--frames-to-process", "-nframes", "--nframes", "-N",
                        nargs=1,
                        action="store",
                        dest="nframes",
                        default=[-1],
                        help="How many frames to process."
                             "Default is all (-1).",)

    parser.add_argument("--from-frame", "-start", "--start",
                        nargs=1,
                        action="store",
                        dest="start",
                        default=[0],
                        help="In which frame to start processing."
                              "Default is 0.",)

    parser.add_argument("--save-plots", "-plot",
                        action="store_true",
                        dest="save_plots",
                        required=False,
                        help="Save processed images. Will slow down the code.")

    parser.add_argument("--plot-path",
                        nargs=1,
                        action="store",
                        dest="plot_path",
                        default=["plot"],
                        required=False,
                        help="Save processed images. Will slow down the code.")

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        required=True,
                        help="Output file with segmentation in csv format.",)

    args = parser.parse_args()

    main()
