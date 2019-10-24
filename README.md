# Introduction

This project aims to provide the information needed to build an ARGUS-like
coastal monitoring system based on single board computers and FLIR
(formally Point Grey) machine vision cameras.

This a continuation and update to the system deployed at the Figure 8 pools
site, which was detailed in [this](https://www.mdpi.com/2072-4292/10/1/11) paper and was operational for over an year.

# 1 Hardware

## 1.1 Computer Board

This project has been developed using a Raspberry Pi Model 3 B. Better results
may be achieved using the new Raspberry Pi 4.

The components of the system are:
1. [Raspberry Pi board](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/)
2. [Raspberry Pi 7in display](https://www.raspberrypi.org/products/raspberry-pi-touch-display/)
3. [Raspberry Pi display case](https://thepihut.com/products/raspberry-pi-7-touchscreen-display-case-white)
4. [16Gb+ SD card](https://www.raspberrypi.org/documentation/installation/sd-cards.md)
5. Keyboard
6. Mouse

**Note:** the 7in display case mentioned in the link above is not compatible with the
Raspberry Pi model 4 B. At the time of writing there seems not to be such a case
for the Pi model 4.

Assembly should be straight forward but if in doubt, watch this video:

[![](doc/SettingupyourRaspberryPi.png)](https://www.raspberrypi.org/help/quick-start-guide/2/)

## 1.2 Machine Vision Camera

Our camera of choice is the [Flea 3 USB3 3.2 MP](https://www.flir.com/products/flea3-usb3/) model. However, the implementation provided here should work with any FLIR machine
vision USB3 camera.

For this project, we used a [Tamron 8mm lens](https://www.flir.fr/products/tamron-8mm-11.8inch-c-mount-lens/). Note that you will need a C to CS mount adaptor if your camera has a CS mount and your lens has a C mount.

# 2 Software

## 2.1 Operating System (OS)

FLIR recommends Ubuntu for working with their cameras. Unfortunately,
the full version of Ubuntu is too demanding to run on the Raspberry Pi.
Therefore, we recommend [Ubuntu Mate](https://www.google.com).

### 2.1.1 Installation

On a separate computer,

1. Download the appropriate Ubuntu Mate image from [here](https://ubuntu-mate.org/raspberry-pi).
2. Use [Fletcher](https://www.balena.io/etcher/) to flash the image to the SD card.

Insert the SD card in the Raspberry Pi and connect to mains power.

If everything worked, you will be greeted by Ubuntu Mate's installer. Simply
follow the installer's instructions and finish the install. If everything goes
correctly, the system will reboot and you will be greeted by the welcome screen.

![](doc/mate_welcome.png)

## 2.2 FLIR's Dependencies

Before installing FLIR's software, there are several package dependencies that
need to be installed.

First update your Ubuntu install:

```bash
sudo apt update
sudo apt dist-upgrade
```
This will take a while to finish. Go grab a coffee.

Next, install the build-essentials package:

```bash
sudo apt install build-essential
```

Now install the required dependencies:

```bash
sudo apt install libusb-1.0-0 libpcre3-dev
```

## 2.3 FLIR Spinnaker Setup

[Spinnaker](https://www.flir.com/products/spinnaker-sdk/) is the software responsible for interfacing the camera and the computer.
Download Spinnaker from [here](https://flir.app.boxcn.net/v/SpinnakerSDK).

Open the folder where you downloaded Spinnaker and decompress the file.

Now, open a terminal in the location of the extracted files and do:
```bash
sudo sh install_spinnaker_arm.sh
```

Follow the instructions in the prompt until the installation is complete.

**Note:** You may fall into a dependency loop here. Pay close attention to
the outputs in the prompt after running the installer. If in trouble, `apt`
can help you:

```bash
sudo apt install -f --fix-missing
```

From FLIR's README file, it is also recommend to increase the size of USB stream
from 2Mb to 1000Mb. To do this do not follow their instructions, they will
not work for Raspberry Pi Based systems. Instead do:

```bash
sudo nano /boot/cmdline.txt
```

Append to the end of the file:

```
usbcore.usbfs_memory_mb=1000
```
Reboot your Raspberry Pi and check if it worked with:

```
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

Connect your camera and launch Spinnaker GUI:

```bash
spinview
```

I everything went well, you should see your camera in the USB Interface panel
on the left.

![](doc/spinview.png)

We will not use Spinview in this project but it is a useful tool to debug your camera. Please check Spinnaker documentation regarding Spinview usage.

## 2.4 PySpin

Python comes preinstalled with Ubuntu Mate. Only make sure you are using python 3.7.

Before installing FLIR's interface, make sure the following dependencies are met:

```bash
python3.7 -m pip install --upgrade --user numpy matplotlib Pillow==5.2.0
```

Download FLIR's python wheel from [https://flir.app.boxcn.net/v/SpinnakerSDK/folder/74731091944](here).

```bash
sudo python3.7 -m pip install spinnaker_python-1.26.0.31-cp37-cp37m-linux_aarch64.whl
```

Finally, install [OpenCV](https://pypi.org/project/opencv-python/).

```bash
sudo python3.7 -m pip install opencv-python opencv-contrib-python
```


# 3 Image Capture Configuration File

To be defined

# 4 Required improvements

1. Add the ability to handle more than one camera
