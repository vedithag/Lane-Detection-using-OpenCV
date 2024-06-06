'''
laneHSV.py

Team: Keval Visaria, Tejarsri Kasturi, Veditha Gudapati, Chirag Dhoka Jain

This script is designed to detect lane markings in video footage using the HSV color space.
It employs color filtering, edge detection, and Hough Line Transform to identify and 
highlight lanes on roads. The script is part of a project aimed at developing advanced 
driver-assistance systems (ADAS) that enhance vehicle safety through automated perception
of road conditions.
'''

import cv2
import numpy as np

class LaneDetector:
    """
    A class that encapsulates lane detection functionality, maintaining a history
    of detected lane lines and providing a smoothed average lane position over time.
    """
    def __init__(self, window_size=10):
        """
        Initializes the LaneDetector with a fixed-size buffer for averaging lanes over time.
        """
        self.window_size = window_size
        self.left_lane_buffer = []
        self.right_lane_buffer = []

    def update(self, left_lane, right_lane):
        """
        Updates the lane buffers with the latest detected lanes and pops out the oldest if necessary.
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
        Averages the lane positions stored in the buffers to obtain a smoother lane detection.

        :return: A tuple containing the averaged left and right lane positions.
        """
        left_lane = np.mean(self.left_lane_buffer, axis=0) if self.left_lane_buffer else None
        right_lane = np.mean(self.right_lane_buffer, axis=0) if self.right_lane_buffer else None
        return left_lane, right_lane

def detect_lanes(img):
    """
    Detects lanes in a given image frame using color filtering, edge detection, 
    and Hough Line transformation to identify lane lines.

    """

    # Convert the image to HSV color space to isolate specific colors (yellow and white).
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for yellow color (common lane marking).
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    # Create a mask for white color (another common lane marking).
    white_mask = cv2.inRange(hsv, (0, 0, 180), (255, 25, 255))

    # Combine the two masks to capture both yellow and white lane markings.
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    # Apply the combined mask to the original image to isolate lane-like features.
    filtered_image = cv2.bitwise_and(img, img, mask=combined_mask)
    cv2.imshow('Filtered Image', filtered_image)  # Display the filtered image for visual feedback.

    # Convert the filtered image to grayscale for further processing.
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Image', gray_image)  # Display the grayscale image.

    # Apply Gaussian blur to reduce noise and smooth the image.
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cv2.imshow('Blurred Image', blurred_image)  # Display the blurred image.

    # Perform adaptive Canny edge detection to highlight edges in the image.
    edges = adaptive_canny(blurred_image)  # `adaptive_canny` applies Canny edge detection.
    cv2.imshow('Canny Edges', edges)  # Display the edges.

    # Define a region of interest (ROI) to focus only on the area where lanes are expected.
    height, width = img.shape[:2]
    # Use a triangular ROI with points at the bottom left, middle top, and bottom right.
    vertices = np.array([[(0, height), (width // 2, height // 3), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)  # Mask edges to only keep the ROI.
    cv2.imshow('Masked Edges', masked_edges)  # Display the masked edges.

    # Use Hough Line Transform to detect lines in the masked edge image.
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 30, minLineLength=30, maxLineGap=100)

    # Fit the detected lines into left and right lanes.
    left_lane, right_lane = fit_lines(lines, height)  # `fit_lines` determines left and right lanes.

    return left_lane, right_lane  # Return the detected left and right lanes (if any).



def fit_lines(lines, height):
    """
    Fits lines detected from Hough transform to construct left and right lane lines.
    """

    # Initialize lists to store parameters of lines on the left and right sides of the lane.
    left_lines = []
    right_lines = []

    # Check if any lines were detected
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Ensure the line has a valid slope (avoid division by zero).
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)  # Calculate the slope of the line.

                    # Intercept can be calculated using the line equation: y = mx + b => b = y - mx
                    intercept = y1 - slope * x1

                    # Filter lines based on the slope to determine if they belong to the left or right lanes.
                    # Negative slopes are left lanes, positive slopes are right lanes.
                    if slope < -0.5:  # A steep negative slope indicates a left lane line.
                        left_lines.append((slope, intercept))
                    elif slope > 0.5:  # A steep positive slope indicates a right lane line.
                        right_lines.append((slope, intercept))

    # Using the collected line parameters, fit a single line for both left and right lanes.
    # This fitting can be done by averaging the parameters of the collected lines.
    left_lane = fit_single_line(left_lines, height)
    right_lane = fit_single_line(right_lines, height)

    return left_lane, right_lane

def fit_single_line(lines, height):
    """
    Averages the lines based on their slopes and intercepts to create a single line
    that best represents the detected lane marking. This line is extrapolated to 
    extend from the bottom of the image to a predefined point upwards.
    """
    if not lines:
        return None

    # Average out the lines' parameters to find the best fit line.
    average_slope, average_intercept = np.mean(lines, axis=0)

    # Using the average slope and intercept, calculate the x-coordinates corresponding
    # to the bottom of the image and a predefined point upwards (here, 65% of the image height).
    y1 = height  # Bottom of the image
    y2 = int(height * 0.65)  # 65% of the image height

    x1 = int((y1 - average_intercept) / average_slope)
    x2 = int((y2 - average_intercept) / average_slope)

    return np.array([[x1, y1, x2, y2]])  # Return the endpoints of the extrapolated line.

def region_of_interest(img, vertices):
    """
    Applies an image mask that only keeps the region of the image defined by the polygon
    formed from 'vertices'. The rest of the image is set to black.
    """
    # Create a mask that is the same size as the image but filled with zeros (black).
    mask = np.zeros_like(img)

    # Fill the area defined by the polygon (vertices) with white (255).
    cv2.fillPoly(mask, vertices, 255)

    # Apply the mask to the input image using a bitwise AND operation,
    # which retains only the parts of the image that are white in the mask.
    return cv2.bitwise_and(img, mask)

def adaptive_canny(image, sigma=0.33):
    """
    Applies the Canny edge detection algorithm with automatic threshold calculation
    using the median pixel value of the image.
    """
    # Calculate the median pixel value of the image
    v = np.median(image)

    # Calculate the lower threshold value as a function of the median value
    lower = int(max(0, (1.0 - sigma) * v))
    # Calculate the upper threshold value as a function of the median value
    upper = int(min(255, (1.0 + sigma) * v))

    # Apply the Canny edge detector with the calculated thresholds
    return cv2.Canny(image, lower, upper)


def draw_lanes(img, lanes):
    '''
    Draws lane lines on an image.

    This function takes an image and a list of lane lines, where each lane line is defined
    by its endpoints, and draws these lines directly onto the image.
    '''
    if lanes:
        for lane in lanes:
            if lane is not None:
                # Convert points into integer tuples
                start_point = (int(lane[0, 0]), int(lane[0, 1]))
                end_point = (int(lane[0, 2]), int(lane[0, 3]))
                # Draw the line using the converted points
                cv2.line(img, start_point, end_point, (0, 255, 0), 10)
    return img

# Load video
cap = cv2.VideoCapture("C:/Users/vedit/OneDrive/Desktop/NEU/PRCV/Final Project/Test/lane-detection-openCV-master/challenge_video.mp4")

# Initialize LaneDetector
lane_detector = LaneDetector(window_size=5)  # Adjust window size for smoother averaging

# Define window size
window_width, window_height = 400, 600  # Adjust dimensions as needed

# Create a named window and resize it
cv2.namedWindow('Lane Detection Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lane Detection Video', window_width, window_height)

# Flags for key events
print_steps = False
display_video = False

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and update lanes
    left_lane, right_lane = detect_lanes(frame)
    lane_detector.update(left_lane, right_lane)
    left_lane, right_lane = lane_detector.get_lanes()

    # Draw and display lanes
    lanes_img = draw_lanes(frame, [left_lane, right_lane])
    combo_image = cv2.addWeighted(frame, 0.8, lanes_img, 1, 0)

    # Check for key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('n'):  # Print output for each preprocessing step
        print_steps = not print_steps
    elif key == ord('v'):  # Display output lane-detected video
        display_video = not display_video

    # Show output for each preprocessing step if requested
    if print_steps:
        cv2.imshow('Lane Detection', combo_image)

    # Display output lane-detected video if requested
    if display_video:
        cv2.imshow('Lane Detection Video', combo_image)

cap.release()
cv2.destroyAllWindows()
