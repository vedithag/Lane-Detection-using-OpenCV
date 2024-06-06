import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def process_and_visualize_lanes(json_path, image_dir):
    with open(json_path, 'r') as file:
        json_gt = [json.loads(line) for line in file]
    
    for item in json_gt:  # Process each entry in the JSON file
        raw_file = item['raw_file']
        img_path = os.path.join(image_dir, raw_file)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Image not found at {img_path}")
            continue

        cv2.imshow('Original Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        gt_lanes = item['lanes']
        y_samples = item['h_samples']
        gt_lanes_vis = [[(x, y) for x, y in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
        img_vis = img.copy()

        for lane in gt_lanes_vis:
            cv2.polylines(img_vis, [np.array(lane, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=5)

        cv2.imshow('Lanes Visualized', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        mask = np.zeros_like(img)
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]]
        for i, lane in enumerate(gt_lanes_vis):
            cv2.polylines(mask, [np.array(lane, dtype=np.int32)], isClosed=False, color=colors[i], thickness=5)

        label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for i, color in enumerate(colors):
            label[(mask == color).all(axis=2)] = i + 1

        plt.imshow(label, cmap='gray')
        plt.title('Label Image')
        plt.show()

# Use absolute paths and ensure they are correct
json_path = 'D:/jsbx/attempt/TUSimple/train_set/label_data_0313.json'
image_dir = 'D:/jsbx/attempt/TUSimple/train_set/'

process_and_visualize_lanes(json_path, image_dir)
