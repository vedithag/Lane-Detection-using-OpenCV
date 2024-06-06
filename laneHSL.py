'''
laneHSL.py

Team: Keval Visaria, Tejarsri Kasturi, Veditha Gudapati, Chirag Dhoka Jain

This script processes video footage to detect and visualize lanes on the road. It uses image processing techniques
such as color conversion to HLS, color thresholding, Gaussian blurring, Canny edge detection, and Hough Line 
Transformation. The script supports interactive display modes, showing raw and processed outputs to aid in tuning
and debugging the lane detection algorithms.

The processing pipeline is designed to handle varying lighting and road conditions by focusing on specific color 
ranges typical for lane markings.
'''

import cv2
import numpy as np

class LaneDetector:
    """
    A class to handle the detection and smoothing of lanes over frames for video streams.
    It maintains a buffer for each detected lane to average over time, providing a more stable detection output.
    """
    def __init__(self, window_size=10):
        """
        Initializes the LaneDetector with a specified buffer size for averaging detected lanes.

        """
        self.window_size = window_size
        self.left_lane_buffer = []
        self.right_lane_buffer = []

    def update(self, left_lane, right_lane):
        """
        Updates the internal buffers with new lane detections.
        """
        if left_lane is not None:
            self.left_lane_buffer.append(left_lane)
            if len(self.left_lane_buffer) > self.window_size:
                self.left_lane_buffer.pop(0)
        if right_lane is not None:
            self.right_lane_buffer.append(right_lane)
            if len(self.right_lane_buffer) > self.window_size:
                self.right_lane_buffer.pop(0)

    def get_lanes(self):
        """
        Averages the buffered lane coordinates to provide a stable output.
        """
        left_lane = np.mean(self.left_lane_buffer, axis=0) if self.left_lane_buffer else None
        right_lane = np.mean(self.right_lane_buffer, axis=0) if self.right_lane_buffer else None
        return left_lane, right_lane

def detect_lanes(img):
    """
    Processes an image to detect lanes using color filtering, edge detection, and Hough Line transform.
    """
    # Convert the BGR image to HLS color space to better detect yellow and white colors.
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Define the color thresholds for yellow and white colors to detect lane lines.
    yellow_lower = np.array([15, 100, 120], dtype=np.uint8)
    yellow_upper = np.array([35, 255, 255], dtype=np.uint8)
    white_lower = np.array([0, 210, 0], dtype=np.uint8)
    white_upper = np.array([180, 255, 255], dtype=np.uint8)

    # Create masks for yellow and white colors within the specified thresholds.
    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hls, white_lower, white_upper)

    # Combine the yellow and white masks to get the final mask.
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    filtered = cv2.bitwise_and(img, img, mask=mask)

    # Convert the filtered image to grayscale and apply Gaussian blur.
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive Canny edge detection to find edges.
    edges = adaptive_canny(blurred)

    # Define a region of interest to focus on the road area.
    height, width = img.shape[:2]
    vertices = np.array([[(0, height), (width // 2, height // 3), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Detect lines using the Hough Line transform.
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=100)

    # Return None if no lines are detected.
    if lines is None:
        return None, None, None, None, None, None, None
    
    # Fit lines to detect left and right lanes.
    left_lane, right_lane = fit_lines(lines, height)
    return left_lane, right_lane, filtered, gray, blurred, edges, masked_edges


def region_of_interest(img, vertices):
    """
    Applies a mask to the image, keeping only the region defined by the vertices polygon.
    """
    mask = np.zeros_like(img)  # Create a mask that's the same size as the image.
    cv2.fillPoly(mask, vertices, 255)  # Fill the specified polygon area with white.
    return cv2.bitwise_and(img, mask)  # Apply the mask to the original image.


def fit_lines(lines, height):
    """
    Separates the detected lines into left and right lanes based on their slopes and
    computes a single line for each lane from the average of these lines.
    """
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 != x1:  # Prevent division by zero.
                slope = (y2 - y1) / (x2 - x1)  # Calculate the slope.
                if slope < -0.5:  # Negative slope for left lane.
                    left_lines.append((slope, y1 - slope * x1))
                elif slope > 0.5:  # Positive slope for right lane.
                    right_lines.append((slope, y1 - slope * x1))
    left_lane = fit_single_line(left_lines, height)
    right_lane = fit_single_line(right_lines, height)
    return left_lane, right_lane


def fit_single_line(lines, height):
    """
    Fits a single line to a set of line segments by averaging their slopes and intercepts.
    """
    if not lines:
        return None
    
    # Compute the mean slope and intercept of the input lines.
    line = np.mean(lines, axis=0)
    slope, intercept = line

    # Define the starting and ending points of the line.
    y_start = height
    y_end = int(height * 0.65)
    x_start = int((y_start - intercept) / slope)
    x_end = int((y_end - intercept) / slope)

    # Return the array containing the start and end points of the fitted line.
    return np.array([[x_start, y_start, x_end, y_end]])


def adaptive_canny(image, sigma=0.33):
    """
    Applies adaptive Canny edge detection to an image.
    """
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def draw_lanes(img, lanes):
    """
    Draws the detected lane lines on the original image.
    """
    lane_img = np.zeros_like(img)
    for lane in lanes:
        if lane is not None:
            start_point = (int(lane[0, 0]), int(lane[0, 1]))
            end_point = (int(lane[0, 2]), int(lane[0, 3]))
            cv2.line(lane_img, start_point, end_point, (0, 255, 0), 10)
    return cv2.addWeighted(img, 0.8, lane_img, 1, 0)


cap = cv2.VideoCapture("C:/Users/vedit/OneDrive/Desktop/NEU/PRCV/Final Project/Test/lane-detection-openCV-master/.mp4")
lane_detector = LaneDetector(window_size=5)

cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lane Detection', 400, 600)

show_preprocessing = False
show_video = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF

    if show_preprocessing:
        # Preprocessing steps
        left_lane, right_lane, filtered, gray, blurred, edges, masked_edges = detect_lanes(frame)
        if left_lane is not None and right_lane is not None:
            combo_image = draw_lanes(frame, [left_lane, right_lane])
            cv2.imshow('Lane Detection', combo_image)
            cv2.imshow('Filtered Image', filtered)
            cv2.imshow('Gray Image', gray)
            cv2.imshow('Blurred Image', blurred)
            cv2.imshow('Edges Image', edges)
            cv2.imshow('Masked Edges Image', masked_edges)
            if key == ord('v'):
                show_video = True
                show_preprocessing = False
            elif key == ord('q'):
                break

    elif show_video:
        left_lane, right_lane = detect_lanes(frame)
        if left_lane is not None and right_lane is not None:
            combo_image = draw_lanes(frame, [left_lane, right_lane])
            cv2.imshow('Lane Detection', combo_image)
            if key == ord('n'):
                show_preprocessing = True
                show_video = False
            elif key == ord('q'):
                break

    else:
        if key == ord('n'):
            show_preprocessing = True
        elif key == ord('v'):
            show_video = True
        elif key == ord('q'):
            break

cv2.waitKey(25)
cap.release()
cv2.destroyAllWindows()
