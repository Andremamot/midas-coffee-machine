#pragma once

/*
 * Camera type
 1. MIPI CSI-2 (e.g. IMX219) - uses GStreamer pipeline with v4l2src
    - Output frames are resized to 1920x1080 for DRP-AI processing
 2. USB camera (e.g. Logitech C920) - uses OpenCV VideoCapture directly
    - Output frames for DRP-AI processing are 640x480 (maximum supported size
 for RZ/V2H)
 */
#define INPUT_CAM_TYPE (2)

#define MIPI_WIDTH (1920)
#define MIPI_HEIGHT (1080)

#define USB_WIDTH (2592)
#define USB_HEIGHT (1944)