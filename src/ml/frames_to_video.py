import argparse

import cv2

from glob import glob
from natsort import natsorted


def main():
    """Call the main script."""
    inp = args.input[0]
    out = args.output[0]
    fps = int(args.fps[0])
    files = natsorted(glob(inp + "/*"))
    if not files:
        print("Verify input frames. Make sure they are png images.")

    frame = cv2.imread(files[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(out,
                            cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (width, height))

    k = 0
    for image in files:
        print(" -- processing {} of {}".format(k+1, len(files)), end="\r")
        video.write(cv2.imread(image))
        k += 1

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':

    print("\nCreating animation, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input path with frames.",)

    parser.add_argument("--frames-per-second", "-fps", "--fps",
                        nargs=1,
                        action="store",
                        dest="fps",
                        default=[60],
                        required=False,
                        help="Frames per second of the output video.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["video.mp4"],
                        required=False,
                        help="Output file name.",)

    args = parser.parse_args()

    main()
