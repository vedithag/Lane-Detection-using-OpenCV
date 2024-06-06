import os
import json
import cv2
import numpy as np

# Path to the TuSimple dataset directory
data_dir = 'TUSimple/train_set/'
img_dir = os.path.join(data_dir, 'clips')

# Function to load images and labels
def load_data(json_files):
    images = []
    labels = []

    for file in json_files:
        with open(os.path.join(data_dir, file)) as f:
            for line in f:
                data = json.loads(line)
                img_path = os.path.join(img_dir, data['raw_file'])
                image = cv2.imread(img_path)
                lanes = data['lanes']
                h_samples = data['h_samples']
                
                label_img = np.zeros_like(image, dtype=np.uint8)
                for lane, h in zip(lanes, h_samples):
                    for x, y in zip(lane, h):
                        if x != -2:  # -2 indicates no lane marking
                            cv2.circle(label_img, (x, y), 5, (255, 0, 0), -1)  # Blue color for lane markings

                images.append(image)
                labels.append(label_img)

    return images, labels

# Example usage
json_files = ['label_data_0601.json']
images, labels = load_data(json_files)
