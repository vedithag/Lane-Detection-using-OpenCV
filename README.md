# Lane Detection using OpenCV
## Overview
This repository contains the implementation of a basic lane detection system using OpenCV in Python. The project demonstrates how to detect and highlight lanes on the road in real-time video feeds, which is a fundamental task in the development of autonomous driving systems.

## Features
Real-time lane detection using webcam or video input.
Utilization of OpenCV functions for image manipulation, such as converting images to grayscale, applying Gaussian blur, and edge detection.
Application of the Hough Transform technique to detect lines in edge-detected images.
Algorithm to extrapolate and draw lane lines based on detected edges.

## Prerequisites
Before you begin, ensure you have met the following requirements:

Python 3.6 or higher
OpenCV library (cv2)
NumPy
Matplotlib (optional for visualization)

## Installation
To install the necessary libraries, you can use pip:
### pip install opencv-python numpy matplotlib

## Usage
To run the lane detection script, you can use the following command:
### python lane_detection.py
Replace lane_detection.py with the path to the script if it is located in a different directory.

## How It Works
The script performs the following steps to detect lanes:

Image Preprocessing: Converts the video feed to grayscale and applies Gaussian blur to smooth the image.
Edge Detection: Uses the Canny edge detector to find edges in the image.
Region of Interest Selection: Isolates the region of the image where lane lines are commonly found.
Line Detection: Applies the Hough Transform to detect lines in the edged image.
Line Combination and Extrapolation: Combines fragmented line segments into complete lane lines and extrapolates them to cover the full lane length.
Result Display: The detected lane lines are overlaid on the original video for visualization.
