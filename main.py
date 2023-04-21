# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the input shape of the image
height = 533  # adjust as necessary
width = 800   # adjust as necessary
channels = 3  # assuming RGB image
input_shape = (height, width, channels)

# Load the input image
file_path = 'Skin_graft.jpg'  # change to your image file path
img_rgb = cv2.imread(file_path)

# Preprocess the image (resize, crop, normalize, etc.) to prepare it for the CNN
# You may need to experiment with different preprocessing techniques depending on your dataset
img_resized = cv2.resize(img_rgb, (width, height))
img_cropped = img_resized[150:650, 250:1050]  # adjust crop area as necessary
img_normalized = img_cropped / 255.0  # normalize pixel values to [0,1]

# Reshape the input image to match the input shape of the CNN
img_reshaped = np.expand_dims(img_normalized, axis=0)

# Create a sequential model
model = Sequential()

# Add convolutional layers to the model
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Add fully connected layers to the model
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test)

# Make predictions on new images
predictions = model.predict(new_images)

# HELLLLLOOOOOOOOO
