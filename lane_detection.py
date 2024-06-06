'''
lane_detection.py

Team: Keval Visaria, Tejasri Kasturi, Veditha Gudapati, Chirag Dhoka Jain

This script contains a class called Lane for performing lane detection on images. It utilizes various computer vision techniques such as adaptive thresholding, perspective transformation, sliding window method for lane line detection, and curvature calculation. The Lane class provides methods for initializing lane parameters, extracting lane lines from images, overlaying detected lane lines on the original image, calculating curvature, and determining the position of the car relative to the center of the lane.

The Lane class includes the following methods:

- __init__: Initializes the lane with an image and sets up various parameters.
- adaptive_threshold: Applies adaptive thresholding to the input frame.
- r_position: Calculates the position of the car relative to the center of the lane.
- calculate_curvature: Calculates the road curvature in meters.
- calculate_histogram: Calculates the image histogram.
- display_curvature_offset: Displays curvature and offset statistics on the image.
- get_lane_line_previous_window: Uses the lane line from the previous sliding window to get the parameters for the polynomial line for filling in the lane line.
- locate_lane_line_indices: Locates lane lines using sliding windows and fits a polynomial to each detected line.
- get_line_markings: Isolates lane lines from the input frame.
- extract_lane_lines: Extracts lane lines from the given frame using various thresholding techniques.
- find_histogram_peaks: Finds the left and right peaks of the histogram.
- overlay_lane_lines: Overlays detected lane lines on the original frame.
- perspective_transform: Performs perspective transformation to obtain a bird's eye view of the lane.
- plot_roi: Plots the region of interest on an image.
- find_lane_line_indices_sliding_windows: Finds the indices of lane line pixels using the sliding windows technique.
- calculate_car_position: Calculates the position of the car relative to the center of the lane.

This script is a fundamental component of a lane detection system and provides functionalities for processing images and detecting lane lines in real-time applications.
'''

import cv2 # Import the OpenCV library to enable computer vision
import numpy as np # Import the NumPy scientific computing library
import edge_detection as edge # Handles the detection of lane lines
import matplotlib.pyplot as plt # Used for plotting and error checking
import os


class Lane:
    def __init__(self, frame):
        """
        Initialize the lane with an image from the specified path.
        """
        self.frame = frame

        # Validate if the image loaded correctly
        if self.frame is None:
            raise FileNotFoundError(f"Image could not be loaded from: {frame}")

        self.original_frame = frame
        self.original_dimensions = self.frame.shape[::-1][1:]

        # Dimensions and horizon line calculation
        width, height = self.frame.shape[1], self.frame.shape[0]
        self.width = width
        self.height = height
        horizon = int(0.6 * height)
        
        self.roi_points = np.float32([
            (width * 0.45, horizon),  # Top-left corner
            (width * 0.1, height),         # Bottom-left corner            
            (width * 0.9, height),         # Bottom-right corner
            (width * 0.55, horizon)])  # Top-right corner
        

        # Adjusted points for perspective transform
        self.side_padding = int(0.25 * width)
        
        self.desired_roi_points = np.float32([
            [self.side_padding, 0], 
            [self.side_padding, height],
            [width - self.side_padding, height],
            [width - self.side_padding, 0]
        ])


        # Image processing placeholders
        self.warped_frame = None
        self.matrix = None
        self.inverse_matrix = None
        self.histogram = None

        # Lane detection sliding window settings
        self.no_of_windows = 10
        self.margin = int((1/12) * width)  # Window width is +/- margin
        self.minpix = int((1/24) * width)  # Min no. of pixels to recenter window
        
        # Lane polynomial coefficients and line positions
        self.left_line_fit = None
        self.right_line_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.plot_y = None
        self.left_fitx = None
        self.right_fitx = None
        self.left_x = None
        self.right_x = None
        self.left_y = None
        self.right_y = None

        # Calibration for pixel to real-world conversion
        self.ym_per_pix = 10.0 / 1000  # meters per pixel vertically
        self.xm_per_pix = 3.7 / 781    # meters per pixel horizontally

        # Curvature and lane offset
        self.left_curvem = None
        self.right_curvem = None
        self.lane_offset = None
    
    def adaptive_threshold(self, frame):
        """Apply adaptive thresholding to the input frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        threshold = 120 + (mean_brightness - 128)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def r_position(self, print_to_terminal=False):
        """Calculate the position of the car relative to the center of the lane"""
        if self.left_fit is None or self.right_fit is None:
            return None

        # Find the x coordinate of the lane line bottom
        height = self.frame.shape[0]
        left_bottom = self.left_fit[0]*height**2 + self.left_fit[1]*height + self.left_fit[2]
        right_bottom = self.right_fit[0]*height**2 + self.right_fit[1]*height + self.right_fit[2]

        center_lane = (right_bottom + left_bottom) / 2
        center_offset = (center_lane - self.frame.shape[1]/2) * self.xm_per_pix * 100

        if print_to_terminal:
            print(f"Center Offset: {center_offset:.2f} cm")
            
        self.lane_offset = center_offset

        return center_offset

    
    def calculate_curvature(self, print_to_terminal=False):
        """
        Calculate the road curvature in meters.

        :param: print_to_terminal Display data to console if True
        :return: Radii of curvature
        """
        # Check if lane line points have been found
        if self.left_y is None or self.left_x is None or self.right_y is None or self.right_x is None:
            print("No lane lines detected, cannot calculate curvature.")
            return None, None  # Return None for both curvatures if no points to use

        # Set the y-value where we want to calculate the road curvature.
        # Select the maximum y-value, which is the bottom of the frame.
        plot_y = np.linspace(0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0])
        y_eval = np.max(plot_y)

        print(f"y_eval: {y_eval}")
        # Fit polynomial curves to the real world environment
        left_fit_cr = np.polyfit(self.left_y * self.ym_per_pix, self.left_x * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.right_y * self.ym_per_pix, self.right_x * self.xm_per_pix, 2)

        # Calculate the radii of curvature
        left_curvem = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.abs(2 * left_fit_cr[0])
        right_curvem = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.abs(2 * right_fit_cr[0])

        # Display on terminal window
        if print_to_terminal:
            print(f"Left Curvature: {left_curvem:.2f} m, Right Curvature: {right_curvem:.2f} m")

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem

        return left_curvem, right_curvem

            
            
    def calculate_histogram(self, frame=None, plot=True):
        """Calculate the image histogram."""

        if frame is None:
            frame = self.warped_frame

        self.histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)
        histogram = self.histogram 

        if plot:
            _, ax = plt.subplots(1, figsize=(10, 5))
            ax.plot(histogram)
            ax.set_title('Histogram')
            ax.set_xlabel('Pixel Column')
            ax.set_ylabel('Pixel Count')
            plt.show()

        return histogram
    
    def display_curvature_offset(self, frame=None, plot=False):
        """Display curvature and offset statistics on the image"""

        if frame is None:
            frame = self.frame.copy()

     # Check if curvature values are valid before attempting to display them
        if self.left_curvem is None or self.right_curvem is None:
            print("Curvature values are not available.")
            curvature_text = "Curve Radius: N/A"
        else:
            average_curvature = (self.left_curvem + self.right_curvem) / 2
            curvature_text = f"Curve Radius: {average_curvature:.2f} m"

        # Check if center offset is valid before attempting to display it
        if self.lane_offset is None:
            print("Center offset is not available.")
            offset_text = "Center Offset: N/A"
        else:
            offset_text = f"Center Offset: {self.lane_offset:.2f} cm"

        # Put the text onto the frame
        cv2.putText(frame, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, offset_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if plot:
            cv2.imshow("Image with Curvature and Offset", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return frame

        
    def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
        """
        Use the lane line from the previous sliding window to get the parameters
        for the polynomial line for filling in the lane line
        :param left_fit: Polynomial function of the left lane line
        :param right_fit: Polynomial function of the right lane line
        :param plot: To display an image or not
        """
        # If there is no fit, there is nothing to do
        if left_fit is None or right_fit is None:
            return

        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - self.margin)) &
            (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + self.margin))
        )
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - self.margin)) &
            (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + self.margin))
        )
        
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        self.left_x = leftx
        self.right_x = rightx
        self.left_y = lefty
        self.right_y = righty

        if lefty.size > 0 and leftx.size > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if righty.size > 0 and rightx.size > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
        self.plot_y = ploty
        self.left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        self.right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        if plot:
            out_img = np.dstack((self.warped_frame, self.warped_frame, self.warped_frame)) * 255
            window_img = np.zeros_like(out_img)
            left_line_pts = np.array([np.transpose(np.vstack([self.left_fitx - self.margin, ploty]))])
            right_line_pts = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx + self.margin, ploty])))])
            cv2.fillPoly(window_img, np.int_([left_line_pts, right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap='gray')
            ax3.imshow(result)
            ax3.plot(self.left_fitx, ploty, color='yellow')
            ax3.plot(self.right_fitx, ploty, color='yellow')
            plt.show()
            plt.show()

    def locate_lane_line_indices(self):
        """
        Locates lane lines using sliding windows and fits a polynomial to each detected line.
        """
        window_height = np.int32(self.transformed.shape[0] / self.num_windows)
        nonzero_points = self.transformed.nonzero()
        nonzeroy = np.array(nonzero_points[0])
        nonzerox = np.array(nonzero_points[1])

        leftx_base, rightx_base = self.determine_histogram_peaks()
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_indices = []
        right_lane_indices = []

        for window in range(self.num_windows):
            win_y_low = self.transformed.shape[0] - (window + 1) * window_height
            win_y_high = self.transformed.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.window_width
            win_xleft_high = leftx_current + self.window_width
            win_xright_low = rightx_current - self.window_width
            win_xright_high = rightx_current + self.window_width

            good_left_indices = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            good_right_indices = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            if len(good_left_indices) > self.min_pixel_count:
                leftx_current = np.int(np.mean(nonzerox[good_left_indices]))
            if len(good_right_indices) > self.min_pixel_count:
                rightx_current = np.int(np.mean(nonzerox[good_right_indices]))

        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        leftx = nonzerox[left_lane_indices]
        lefty = nonzeroy[left_lane_indices]
        rightx = nonzerox[right_lane_indices]
        righty = nonzeroy[right_lane_indices]

        self.left_coefficients = np.polyfit(lefty, leftx, 2) if leftx.size and lefty.size else None
        self.right_coefficients = np.polyfit(righty, rightx, 2) if rightx.size and righty.size else None

        return self.left_coefficients, self.right_coefficients

    def get_line_markings(self, frame=None):
        """
        Isolates lane lines.

        :param frame: The camera frame that contains the lanes we want to detect
        :return: Binary image containing the lane lines
        """
        if frame is None:
            frame = self.frame

        # Convert the frame to HLS color space
        hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        cv2.imshow('hls_frame', hls_frame)
        
        # Perform Sobel edge detection on the L (lightness) channel
        l_channel = hls_frame[:, :, 1]
        _, sxbinary = cv2.threshold(l_channel, 120, 255, cv2.THRESH_BINARY)
        sxbinary = cv2.GaussianBlur(sxbinary, (3, 3), 0)
        
        # Compute Sobel magnitude separately for x and y
        sobelx = cv2.Sobel(sxbinary, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(sxbinary, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        sxbinary = np.uint8(magnitude)

        # Perform binary thresholding on the S (saturation) channel
        s_channel = hls_frame[:, :, 2]
        _, s_binary = cv2.threshold(s_channel, 80, 255, cv2.THRESH_BINARY)

        # Perform binary thresholding on the R (red) channel
        r_channel = frame[:, :, 2]
        _, r_thresh = cv2.threshold(r_channel, 120, 255, cv2.THRESH_BINARY)

        # Combine the thresholded images
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)
        lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary)

        self.lane_line_markings = lane_line_markings
        return lane_line_markings


    def extract_lane_lines(self, frame=None):
        """
        Extracts lane lines from the given frame.
    
        :param frame: The camera frame containing the lanes to detect.
        :return: Binary image containing the lane lines.
        """
        if frame is None:
            frame = self.frame

        # Use the adaptive threshold method to get initial binary image
        binary_output, _, _ = self.adaptive_threshold(frame)

        # Convert the frame from BGR to HLS color space
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # Perform Sobel edge detection on the L channel
        _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
        sxbinary = edge.blur_gaussian(sxbinary, ksize=3)
        sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

        # Perform binary thresholding on the S channel
        s_channel = hls[:, :, 2]
        _, s_binary = edge.threshold(s_channel, (80, 255))

        # Perform binary thresholding on the R channel
        _, r_thresh = edge.threshold(frame[:, :, 2], thresh=(120, 255))

        # Bitwise AND operation to combine S and R channel thresholds
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        # Combine possible lane lines with possible lane line edges
        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))

        return self.lane_line_markings

    def find_histogram_peaks(self):
        """
        Find the left and right peaks of the histogram.

        Returns the x-coordinate of the left histogram peak and the right histogram peak.
        """
        midpoint = np.int32(self.histogram.shape[0] / 2)
        left_peak = np.argmax(self.histogram[:midpoint])
        right_peak = np.argmax(self.histogram[midpoint:]) + midpoint

        # Return the x-coordinate of the left peak and right peak
        return left_peak, right_peak

    def overlay_lane_lines(self, plot=False):
        """
        Overlay lane lines on the original frame.

        :param plot: Plot the lane lines if True.
        :return: Image with lane overlay.
        """
        # Create a zero array with the same dimensions as the warped frame
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Check if fits exist before trying to use them
        if self.left_fitx is not None and self.right_fitx is not None:
            
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.plot_y]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.plot_y])))])
            pts = np.hstack((pts_left, pts_right))
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        else:
            print("No lane lines to overlay.")
            return self.frame  # Return original frame if no lines to draw

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        new_warp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix,
                                        (self.frame.shape[1], self.frame.shape[0]))
        
        # Combine the result with the original image
        result = cv2.addWeighted(self.frame, 1, new_warp, 0.3, 0)
        
        if plot:
            # Plot the figure
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title("Original Frame With Lane Overlay")
            plt.show()
        
        return result

            
    def perspective_transform(self, frame=None, plot=False):
        """
        Perform perspective transformation.

        :param frame: Current frame.
        :param plot: Plot the warped image if True.
        :return: Bird's eye view of the current lane.
        """
        if frame is None:
            frame = self.lane_line_markings

        height, width = self.frame.shape[:2]

        # Define the desired points for the region of interest
        desired_roi_points = np.float32([
            [0, 0],                 # Top-left corner
            [0, height],            # Bottom-left corner
            [width, height],        # Bottom-right corner
            [width, 0]              # Top-right corner
        ])

        # Calculate the transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(self.roi_points, desired_roi_points)

        # Calculate the inverse transformation matrix
        inv_transformation_matrix = cv2.getPerspectiveTransform(desired_roi_points, self.roi_points)

        # Perform the transform using the transformation matrix
        warped_frame = cv2.warpPerspective(frame, transformation_matrix, self.original_dimensions, flags=cv2.INTER_LINEAR)

        # Convert image to binary
        _, binary_warped = cv2.threshold(warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped  # Assign the transformed frame back to self.warped_frame

        # Display the perspective transformed (i.e. warped) frame
        if plot:
            # Draw the region of interest on the warped image
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, [np.int32(desired_roi_points)], True, (147, 20, 255), 3)

            # Display the image
            cv2.imshow('Warped Image', warped_plot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Update transformation matrices
        self.transformation_matrix = transformation_matrix
        self.inv_transformation_matrix = inv_transformation_matrix

        return self.warped_frame


        
    def plot_roi(self, frame=None, plot=False):
        """
        Plot the region of interest on an image.
        """
        if not plot:
            return

        if frame is None:
            frame = self.frame.copy()

        # print(self.roi_points)
        
        # Overlay trapezoid on the frame
        roi_image = cv2.polylines(frame, np.int32([self.roi_points]), True, (147,20,255), 3)

        # Display the image
        cv2.imshow('ROI Image', roi_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_lane_line_indices_sliding_windows(self, plot=False):
        """
        Finds the indices of lane line pixels using the sliding windows technique.

        :param plot: If True, displays visualization plots.
        :return: Best fit lines for the left and right lines of the current lane.
        """
        # Define sliding window parameters
        margin = self.margin
        frame_sliding_window = self.warped_frame.copy()
        window_height = np.int32(self.warped_frame.shape[0] / self.no_of_windows)
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Find starting points for the sliding windows based on histogram peaks
        leftx_base, rightx_base = self.find_histogram_peaks()
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        # Iterate through each window
        for window in range(self.no_of_windows):
            # Define window boundaries
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_base - margin
            win_xleft_high = leftx_base + margin
            win_xright_low = rightx_base - margin
            win_xright_high = rightx_base + margin

            # Draw windows on the visualization image
            cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low),
                        (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low),
                        (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify nonzero pixels within the window
            good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

            # Append indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # Update window position for the next iteration if enough pixels found
            if len(good_left_inds) > self.minpix:
                leftx_base = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_base = np.int32(np.mean(nonzerox[good_right_inds]))

        # Concatenate arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each lane line
        left_fit, right_fit = None, None
        if leftx.size and lefty.size:
            left_fit = np.polyfit(lefty, leftx, 2)
        if rightx.size and righty.size:
            right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        if plot:
            # Create x and y values for plotting
            ploty = np.linspace(0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            # Generate an image to visualize the result
            out_img = np.dstack((frame_sliding_window, frame_sliding_window, frame_sliding_window)) * 255

            # Colorize the detected lane line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Plot the figure with sliding windows and detected lane lines
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            fig.set_size_inches(10, 10)
            fig.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(frame_sliding_window, cmap='gray')
            ax3.imshow(out_img)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame with Sliding Windows")
            ax3.set_title("Detected Lane Lines with Sliding Windows")
            plt.show()

        return self.left_fit, self.right_fit
    
    
    def calculate_car_position(self, print_to_terminal=False):
        """
        Calculate the position of the car relative to the center

        :param print_to_terminal: Display data to console if True
        :return: Offset from the center of the lane
        """
        # Check if polynomial fits have been computed
        if self.left_fit is None or self.right_fit is None:
            print("No lane lines detected, cannot calculate car position.")
            return None  # Return None if no lane line data

        # Assume the camera is centered in the image.
        car_location = self.frame.shape[1] / 2

        # Find the x coordinate of the lane line bottom
        height = self.frame.shape[0]
        bottom_left = self.left_fit[0]*height**2 + self.left_fit[1]*height + self.left_fit[2]
        bottom_right = self.right_fit[0]*height**2 + self.right_fit[1]*height + self.right_fit[2]

        center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        center_offset = (car_location - center_lane) * self.xm_per_pix * 100

        if print_to_terminal:
            print(f"Center Offset: {center_offset:.2f} cm")

        self.lane_offset = center_offset

        return center_offset