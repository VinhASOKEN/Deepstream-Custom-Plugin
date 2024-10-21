################################################################################
# Copyright (c) 2017-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX:= g++
SRCS:= gstnvinferlpr.cpp  gstnvinferlpr_allocator.cpp gstnvinferlpr_property_parser.cpp \
       gstnvinferlpr_meta_utils.cpp gstnvinferlpr_impl.cpp
INCS:= $(wildcard *.h)
LIB:=/usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream/libnvdsgst_inferlpr.so

NVDS_VERSION:=6.3

CFLAGS+= -fPIC -std=c++11 -DDS_VERSION=\"6.3.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes \
	 -I ../gst-nvdspreprocess/include \
	 -I ../../libs/nvdsinfer \
	 -I ../nvdsinfer \
	 -I /opt/nvidia/deepstream/deepstream-6.3/sources/includes \
	 -DNDEBUG

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_customhelper -lnvdsgst_meta -lnvds_meta \
       -lnvds_infer -lnvbufsurface -lnvbufsurftransform -ldl -lpthread \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)