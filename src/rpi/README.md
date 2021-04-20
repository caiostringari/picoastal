

python

```
sudo apt install python-backports.functools-lru-cache
sudo python3 -m pip install opencv-python opencv-contrib-python
sudo python3 -m pip install numpy matplotlib natsort "picamera[array]" ipython
```

latest x264  and ffmpeg

```
cd ~
sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev  libqtgui4  libqt4-test
git clone --depth 1 https://code.videolan.org/videolan/x264
cd x264/
./configure --host=arm-unknown-linux-gnueabi --enable-static --disable-opencl
make -j3
sudo make install

cd ~
git clone git://source.ffmpeg.org/ffmpeg --depth=1
cd ffmpeg/
./configure --arch=armel --target-os=linux --enable-gpl --enable-libx264 --enable-nonfree --extra-ldflags="-latomic"
make -j3
sudo make install
```
