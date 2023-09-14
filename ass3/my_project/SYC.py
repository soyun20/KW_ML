# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bx4ePEjBsogGFwug3B5a7F0k_VKVxa6t
"""

import os
import urllib.request
import tarfile

# download datasets as a tar.gz file
url = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
filename = 'food-101.tar.gz'
urllib.request.urlretrieve(url, filename)

# Decompression
tar = tarfile.open(filename, 'r:gz')
tar.extractall()
tar.close()

# Delete compressed files
os.remove(filename)

import os
# Controls the output of the TensorFlow log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Keras Module
import keras.backend as K
from keras.layers import Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD

# Importing modules for copying files and other operations
from shutil import copy
from collections import defaultdict
import os

# Function to prepare training set and test set
def prepare_data(filepath, src, dest):
    # Create a dictionary to store the images for each class
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        # Read the file paths and store them in a list
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            # Split the path into class and image name
            food = p.split('/')
            # Append the image name with '.jpg' extension to the corresponding class in the dictionary
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        print("\nCopying images into", food)
        if not os.path.exists(os.path.join(dest, food)):
            # Create a directory for the class if it doesn't exist
            os.makedirs(os.path.join(dest, food))
        for i in classes_images[food]:
            # Copy the images from the source directory to the destination directory
            copy(os.path.join(src, food, i), os.path.join(dest, food, i))
    print("Copying Done!")

# Prepare the training and test sets
prepare_data('food-101/meta/train.txt', 'food-101/images', 'food-101/train')
prepare_data('food-101/meta/test.txt', 'food-101/images', 'food-101/test')

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Input, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Define the identity block of ResNet
def identity_block(X, filters, kernel_size):
    # Store the input value to add it later to the output
    X_shortcut = X

    # Convolution layer, Batch Normalization layer, Activation function layer
    X = Conv2D(filters, kernel_size, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Convolution layer, Batch Normalization layer, Activation function layer
    X = Conv2D(filters, kernel_size, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Convolution layer, Batch Normalization layer
    X = Conv2D(filters, kernel_size, padding='same')(X)
    X = BatchNormalization()(X)

    # Add the output of the shortcut path to the main path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# Define the convolutional block of ResNet
def convolutional_block(X, filters, kernel_size):
    # Store the input value to add it later to the output
    X_shortcut = X

    # Convolution layer, Batch Normalization layer, Activation function layer
    X = Conv2D(filters, kernel_size, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Convolution layer, Batch Normalization layer, Activation function layer
    X = Conv2D(filters, kernel_size, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Convolution layer, Batch Normalization layer
    X = Conv2D(filters, kernel_size, padding='same')(X)
    X = BatchNormalization()(X)

    # Shortcut connection - Perform convolution on the shortcut path
    X_shortcut = Conv2D(filters, kernel_size, padding='same')(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)

    # Add the output of the shortcut path to the main path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# Define the ResNet20 model
def ResNet20(input_shape=(32, 32, 3), classes=101):
    # Input tensor
    X_input = Input(input_shape)
    X = X_input

    # Convolution layer, Batch Normalization layer, Activation function layer, Max pooling layer
    X = Conv2D(64, (3, 3), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(2, 2, padding='same')(X) # output size: (16, 16, 64)

    # convolutional_block, identity_block, Max pooling layer
    X = convolutional_block(X, 128, (3,3))
    X = identity_block(X, 128, (3,3))
    X = MaxPooling2D(2, 2, padding='same')(X) # output size: (8, 8, 128)

    # convolutional_block, identity_block, Max pooling layer
    X = convolutional_block(X, 256, (3,3))
    X = identity_block(X, 256, (3,3))
    X = MaxPooling2D(2, 2, padding='same')(X) # output size: (4, 4, 256)

    # convolutional_block, identity_block, Max pooling layer
    X = convolutional_block(X, 512, (3,3))
    X = identity_block(X, 512, (3,3))
    X = MaxPooling2D(2, 2, padding='same')(X) # output size: (2, 2, 512)

    # global average pooling
    X = GlobalAveragePooling2D()(X)
    # Fully connected layer with softmax activation
    X = Dense(classes, activation='softmax')(X) # output size: (1, 1)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='ResNet20')

    return model

K.clear_session()

n_classes = 101 # Number of output classes
img_width, img_height = 32, 32 # Input image dimensions
train_data_dir = 'food-101/train' # Directory for training data
validation_data_dir = 'food-101/test' # Directory for validation data
nb_train_samples = 75750 # Number of training samples
nb_validation_samples = 25250 # Number of validation samples
batch_size = 32 # Batch size for training

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=10,  # Image rotation
    width_shift_range=0.1,  # Horizontal image shift
    height_shift_range=0.1,  # Vertical image shift
    fill_mode='nearest'  # Image filling mode
    )

# Data preprocessing for validation
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generate training data batches
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Generate validation data batches
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Create ResNet20 model
model = ResNet20(input_shape=(img_height, img_width, 3), classes=n_classes)

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks for saving the best model, logging, and learning rate scheduling
checkpointer = ModelCheckpoint(filepath='best_model_resnet20.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('history_resnet20.log')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00001)  # Learning rate scheduling

# Train the model
history = model.fit(train_generator,
                              steps_per_epoch=nb_train_samples // batch_size,
                              validation_data=validation_generator,
                              validation_steps=nb_validation_samples // batch_size,
                              epochs=20,
                              verbose=1,
                              callbacks=[csv_logger, checkpointer, reduce_lr])

# Save the trained model
model.save('model_trained_resnet20.h5')

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt

# Create subplots for loss and accuracy
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

# Plot training and validation loss
loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

# Plot training and validation accuracy
acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')

# Set labels for the axes
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

# Set legends for the plots
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

# Display the plot
plt.show()