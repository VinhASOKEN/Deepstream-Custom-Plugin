# Deepstream-Custom-Plugin

This repository contains two custom DeepStream plugins designed for enhanced license plate recognition and display:

## 1. License Plate Alignment Infer Plugin

### Overview
This feature is integrated into the existing `gst-nvinfer` plugin of DeepStream. I named it `gst-nvinferlpr`. It aligns license plate images that have been cropped by a detection model using specific landmark points before they are processed for inference. This alignment helps improve the accuracy of the recognition process.

### How It Works
1. During the inference process, cropped images of a license plate are aligned before being input to the recognition model.
2. The alignment process adjusts the orientation of the image based on these landmark points.
3. The aligned image is subsequently processed by the inference model, enhancing recognition accuracy.

## 2. License Plate Recognition Display Plugin

### Overview
This custom plugin filters and maintains the display of license plate recognition results that match the correct format. It validates each recognition result against predefined patterns specified in a configuration file.

### How It Works
1. The plugin receives the recognized text of a license plate.
2. It verifies the format against patterns defined in a configuration file.
3. Only results that conform to the specified format are displayed, the wrong results are replaced by the correct.

## Getting Started

### Prerequisites
- DeepStream SDK 6.3
- CUDA Toolkit 11.2
- NVIDIA GPU

### Installation
1. Clone the repository
2. Flowing step by step in instruction.h.
