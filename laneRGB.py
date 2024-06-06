"""
laneHSV.py

Team: Keval Visaria, Tejarsri Kasturi, Veditha Gudapati, Chirag Dhoka Jain

This script is designed to process video footage to detect and visualize lane lines. 
It uses image processing techniques including color space conversion, masking, 
Gaussian blurring, Canny edge detection, and Hough Line Transformation to identify 
and highlight the lanes on the road. The script supports interactive viewing of 
processing steps and the final output.

"""
import cv2
import numpy as np

class LaneDetector:
    """
    Maintains a buffer for detected lanes to smooth out fluctuations in lane detection over time.
    This helps stabilize the detection output across consecutive frames.
    """
    def __init__(self, window_size=10):
        """
        Initializes the LaneDetector with a buffer size that defines how many frames' data 
        to keep for smoothing purposes.
        """
        self.window_size = window_size
        self.left_lane_buffer = []
        self.right_lane_buffer = []

    def update(self, left_lane, right_lane):
        """
        Updates the buffers with newly detected lanes and ensures the buffer doesn't exceed 
        the predefined size.
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
        Returns the average of the stored lane detections to get a smoothed estimation of the lane.
        """
        left_lane = np.mean(self.left_lane_buffer, axis=0) if self.left_lane_buffer else None
        right_lane = np.mean(self.right_lane_buffer, axis=0) if self.right_lane_buffer else None
        return left_lane, right_lane

def detect_lanes(img):
    """
    Processes the image to detect lanes. This includes color filtering, grayscale conversion, 
    blurring, edge detection, and applying a mask for region of interest before using Hough 
    Lines to detect lines that represent lane boundaries.
    """
    
    # Convert image to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define yellow and white color ranges in RGB
    yellow_lower = np.array([150, 150, 0], dtype=np.uint8)
    yellow_upper = np.array([255, 255, 100], dtype=np.uint8)
    white_lower = np.array([200, 200, 200], dtype=np.uint8)
    white_upper = np.array([255, 255, 255], dtype=np.uint8)
    
    # Create masks for yellow and white colors
    yellow_mask = cv2.inRange(rgb, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(rgb, white_lower, white_upper)
    
    # Combine masks
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    filtered = cv2.bitwise_and(img, img, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Canny edge detector
    edges = adaptive_canny(blurred)
    
    # Define dynamic ROI based on the new resolution
    height, width = img.shape[:2]
    vertices = np.array([[(0, height), (width // 2, height // 3), (width, height)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    # Adjust Hough transform parameters for lower resolution
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=100)
    left_lane, right_lane = fit_lines(lines, height)

    return left_lane, right_lane


def fit_lines(lines, height):
    """
    Calculates lines from the detected edges using the Hough Transform output. Segregates these lines
    into left and right lanes based on their slope.
    """
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    if slope < -0.5:
                        left_lines.append((slope, y1 - slope * x1))
                    elif slope > 0.5:
                        right_lines.append((slope, y1 - slope * x1))
    left_lane = fit_single_line(left_lines, height)
    right_lane = fit_single_line(right_lines, height)
    return left_lane, right_lane

def fit_single_line(lines, height):
    """
    Averages multiple detected lines to compute a single line that best represents them.
    This function takes all the detected lines, calculates their average slope and intercept,
    and then uses these to compute the endpoints of a single, average line that extends across
    a predetermined portion of the image.
    """
    if not lines:
        return None
    line = np.mean(lines, axis=0)  # Calculate the mean slope and intercept.
    slope, intercept = line
    y_start = height  # Start at the bottom of the frame.
    y_end = int(height * 0.7)  # End the line higher up on the image.
    x_start = int((y_start - intercept) / slope)  # Calculate the x coordinate for y_start.
    x_end = int((y_end - intercept) / slope)  # Calculate the x coordinate for y_end.
    return np.array([[x_start, y_start, x_end, y_end]])  # Return the endpoints of the line.


def region_of_interest(img, vertices):
    """
    Applies a mask to the image that isolates the region of interest (ROI) defined by 'vertices'.
    This function creates a mask that keeps the region inside the vertices and sets everything
    else to black, which helps to focus the lane detection algorithms on specific areas of the image.
    """
    mask = np.zeros_like(img)  # Create a mask that's the same size as the image but is all black.
    cv2.fillPoly(mask, vertices, 255)  # Fill the specified polygon area with white.
    return cv2.bitwise_and(img, mask)  # Return the image where the mask is applied.


def adaptive_canny(image, sigma=0.33):
    """
    Performs Canny edge detection using automatically calculated threshold values.
    This adaptive method uses the median of the image's pixel intensities to set the
    thresholds, making it more robust to different lighting conditions and contrast levels.
    """
    v = np.median(image)  # Find the median of the pixel intensities.
    lower = int(max(0, (1.0 - sigma) * v))  # Set the lower threshold.
    upper = int(min(255, (1.0 + sigma) * v))  # Set the upper threshold.
    return cv2.Canny(image, lower, upper)  # Apply the Canny edge detector and return the result.


def draw_lanes(img, lanes):
    """
    Draws lanes on an image given the coordinates of the lane lines. This function enhances
    the visual feedback by overlaying detected lanes on the original image.
    """
    lane_img = np.zeros_like(img)  # Start with a blank image that matches the input image.
    if lanes:
        for lane in lanes:
            if lane is not None:
                start_point = (int(lane[0, 0]), int(lane[0, 1]))  # Convert to integer tuple for start.
                end_point = (int(lane[0, 2]), int(lane[0, 3]))  # Convert to integer tuple for end.
                cv2.line(lane_img, start_point, end_point, (0, 255, 0), 10)  # Draw the lane with a green line.
    return lane_img  # Return the image with the lanes drawn.



# Load video
cap = cv2.VideoCapture("c:/Users/vedit/OneDrive/Desktop/NEU/PRCV/Final Project/Test/lane-detection-openCV-master/challenge_video.mp4")

# Initialize LaneDetector
lane_detector = LaneDetector(window_size=5)  # Adjust window size for smoother averaging

cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Lane Detection', 800, 600)  # Resize window as needed

# Flag to indicate when to show the output video
show_video = False

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    if not show_video:
        # Detect and update lanes
        left_lane, right_lane = detect_lanes(frame)
        lane_detector.update(left_lane, right_lane)
        left_lane, right_lane = lane_detector.get_lanes()

        # Draw and display lanes
        lanes_img = draw_lanes(frame, [left_lane, right_lane])
        combo_image = cv2.addWeighted(frame, 0.8, lanes_img, 1, 0)
        cv2.imshow('Lane Detection', combo_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            # Preprocessing steps
            # Convert image to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Define yellow and white color ranges in RGB
            yellow_lower = np.array([150, 150, 0], dtype=np.uint8)
            yellow_upper = np.array([255, 255, 100], dtype=np.uint8)
            white_lower = np.array([200, 200, 200], dtype=np.uint8)
            white_upper = np.array([255, 255, 255], dtype=np.uint8)
            
            # Create masks for yellow and white colors
            yellow_mask = cv2.inRange(rgb, yellow_lower, yellow_upper)
            white_mask = cv2.inRange(rgb, white_lower, white_upper)
            
            # Combine masks
            mask = cv2.bitwise_or(yellow_mask, white_mask)
            filtered = cv2.bitwise_and(frame, frame, mask=mask)

            # Convert to grayscale
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Adaptive Canny edge detector
            edges = adaptive_canny(blurred)
            
            # Define dynamic ROI based on the new resolution
            height, width = frame.shape[:2]
            vertices = np.array([[(0, height), (width // 2, height // 3), (width, height)]], dtype=np.int32)
            masked_edges = region_of_interest(edges, vertices)
            
            # Display each preprocessing step
            cv2.imshow('Filtered Image', cv2.resize(filtered, (400, 300)))
            cv2.imshow('Gray Image', cv2.resize(gray, (400, 300)))
            cv2.imshow('Blurred Image', cv2.resize(blurred, (400, 300)))
            cv2.imshow('Edges Image', cv2.resize(edges, (400, 300)))
            cv2.imshow('Masked Edges Image', cv2.resize(masked_edges, (400, 300)))

        elif key == ord('v'):
            show_video = True

    else:
        # Show the output video with detected lanes
        cv2.imshow('Lane Detection', combo_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
