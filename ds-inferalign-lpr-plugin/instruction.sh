export CUDA_VER=12.1
export NVDS_VERSION=6.3

cd /
git clone https://github.com/opencv/opencv.git
mkdir -p build && cd build
cmake ../opencv
make -j4
make install
mv /usr/local/include/opencv4/* /usr/local/include

cp -r /build/lib/* /usr/local/cuda-12.1/lib64

cp /ds-inferalign-lpr-plugin/src/nvdsinfer.h /opt/nvidia/deepstream/deepstream/sources/includes/nvdsinfer.h

cd /nvdsinfer_custom_impl_Yolov8
make

cd /gst-nvinferlpr
make

cd /nvdsinfer
make
