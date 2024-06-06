# train_model.py

import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

# Import U-Net model from unet.py
from unet import unet_model
from load_data import load_images_and_labels, split_data
from preprocessing import preprocess_data

class TrainingPrint(Callback):
    """
    Custom callback to print updates at the beginning and end of each epoch.
    """
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch+1}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch+1}")
        print(f"Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
        print(f"Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")

def train_unet(base_path, json_path, epochs=10, batch_size=8, learning_rate=1e-4):
    # Loading and preprocessing data
    images, labels = load_images_and_labels(base_path, json_path)
    images, labels = preprocess_data(images, labels)
    X_train, X_val, y_train, y_val = split_data(images, labels, test_size=0.2)

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)

    # Print shapes and data types for debugging
    print("Training data shape:", X_train.shape, "Labels shape:", y_train.shape)
    print("Validation data shape:", X_val.shape, "Validation labels shape:", y_val.shape)

    # Building the U-Net model
    input_shape = X_train[0].shape
    model = unet_model(input_shape=input_shape, num_classes=1)  # Assuming binary classification for lane detection

    # Compiling the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Model checkpoint to save the best model
    checkpoint_callback = ModelCheckpoint('unet_lane_detection.keras', save_best_only=True, monitor='val_loss', mode='min')

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback]
    )

    # Save the final model
    model.save('unet_lane_detection_final.h5')


if __name__ == "__main__":
    base_path = 'TUSimple/train_set/'  # Base directory containing the images
    json_path = 'TUSimple/train_set/label_data_0313.json'  # Path to the JSON file with annotations
    train_unet(base_path, json_path)
