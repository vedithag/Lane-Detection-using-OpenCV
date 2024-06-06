'''
main.py

Team: Keval Visaria, Tejashri Kasturi, Veditha Gudapati, Chirag Dhoka Jain

This script performs lane detection on an input image. It loads an image,
identifies lane markings using computer vision techniques, and then applies
various transformations and algorithms to detect and visualize lane lines,
including thresholding, perspective transformation, histogram calculation,
sliding window method for lane line detection, curvature calculation,
center offset calculation, and overlaying detected lane lines on the original image.
It finally saves the processed image in the specified output directory.

 '''

import cv2
import os
import lane_detection  # Import the lane_detection module


root = 'Test_data/test_images/'
FILENAME = '1.jpg'

image = os.path.join(root, FILENAME )

def main():
    # Load a frame 
    original_frame = cv2.imread(image)

    # Create a Lane object
    lane_obj = lane_detection.Lane(original_frame)

    # Perform thresholding to isolate lane lines
    lane_line_markings = lane_obj.get_line_markings()

    # Plot the region of interest on the image
    lane_obj.plot_roi(plot=True)

    # Perform the perspective transform to generate a bird's eye view
    # If Plot == True, show image with new region of interest
    warped_frame = lane_obj.perspective_transform(frame=lane_line_markings, plot=True)

    # Generate the image histogram to serve as a starting point
    # for finding lane line pixels
    histogram = lane_obj.calculate_histogram(plot=True)

    # Find lane line pixels using the sliding window method
    left_fit, right_fit = lane_obj.find_lane_line_indices_sliding_windows(plot=True)

    # Fill in the lane line
    lane_obj.q(left_fit, right_fit, plot=True)

    # Overlay lines on the original frame
    frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=True)

    # Calculate lane line curvature
    left_curvature, right_curvature = lane_obj.calculate_curvature(print_to_terminal=True)

    print("Left Curve", left_curvature)
    print("Right Curve", right_curvature)
    
    # Calculate center offset
    center_offset = lane_obj.calculate_car_position(print_to_terminal=True)

    # Display curvature and center offset on image
    frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=True)

    output_directory = 'Output/'
    # Extract the filename
    size = len(FILENAME)
    new_filename = FILENAME[:size - 4]
    new_filename = os.path.join(output_directory, new_filename + '_thresholded.jpg')

    # Save the new image in the specified directory
    cv2.imwrite(new_filename, lane_line_markings)

    # Display the image
    cv2.imshow("Image", lane_line_markings)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
