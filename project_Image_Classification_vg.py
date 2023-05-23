# -*- coding: utf-8 -*-
"""
Created on Sun May 14 00:36:05 2023

@author: vamsh
"""

# pip install astroNN
import pandas
import numpy
from matplotlib import pyplot as plt
# h5py package allows us to interact with HDF5 (Hierarchical Data Format) files. 
# HDF5 is a file format designed for storing and managing large amounts of numerical data and metadata.
import h5py
# Importing tools from Scikit-learn a popular Machine Learning library
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# Importing tools from the specific implementation of the Keras API integrated within TensorFlow.
# tensorflow.keras serves as the primary API for developing neural networks.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg16 import VGG16
# Setting a random seed ensures that the random number generation during model training is deterministic. 
# This means that running the code with the same seed will produce the same results each time.
from numpy.random import seed
seed(20231405)
tf.random.set_seed(20231405)

#### Loading the Galaxy 10 Dataset
# Extracing the images and labels from data file
with h5py.File('C:\\Users\\vamsh\\Documents\\Uni\\UChicago\\Spring 2023\\MSCA 31009_IP01 - Machine Learning & Predictive Analytics\\Project\\Data\\Galaxy10.h5', 'r') as F:
     images = numpy.array(F['images'])
     labels = numpy.array(F['ans'])

print("Shape of Original Images:\n", images.shape)
print("")
print("Shape of Original Labels:\n", labels.shape)

#### Feature Engineering and Transformations:
# Splitting the data into Training and Test sets:
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state = 20231405)
# Splitting Test set into Validation set for implementing early stopping and fine tuning the model's 
# performance during the training process
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state = 20231405)

# Normalizing training data by dividing by 69 so to expedite the convergence of the model during training
# values will range from 0-1 instead of 0-69
X_train = X_train / 69
X_val = X_val / 69
X_test = X_test / 69

print("Shape of Images for model fitting:\n", X_train.shape, X_val.shape, X_test.shape)
print("")
print("Shape of Images for prediction:\n", y_train.shape, y_val.shape, y_test.shape)
# len(numpy.unique(y_test))

#### Exploratory Data Analysis

# Plot of the first 16 stellar images
features = ['Disk, Face-on, No Spiral', 'Smooth, Completely round', 'Smooth, in-between round', 'Smooth, Cigar shaped', 'Disk, Edge-on, Rounded Bulge', 'Disk, Edge-on, Boxy Bulge', 
            'Disk, Edge-on, No Bulge','Disk, Face-on, Tight Spiral', 'Disk, Face-on, Medium Spiral', 'Disk, Face-on, Loose Spiral']

# 4 by 4 plot of the galaxy Images
fig = plt.figure(figsize=(10, 10)) 
for i in range(16):
    plt.subplot(4,4,i+1)    
    plt.imshow(X_train[i])
    plt.title(features[y_train[i]])
    fig.tight_layout(pad=1.5)
plt.show()

# Plotting the Category Distribution
df = pandas.DataFrame(data=labels)
counts = df.value_counts().sort_index()
# print(counts)
fig, ax = plt.subplots()
ax.bar(features, counts)
ax.set_xticklabels(features, rotation=90)
plt.title("Class Distribution of Galaxy10 Dataset")
plt.show()

#### Proposed Approaches (Models)

########
# Baseline CNN model (not pre-trained so no Transfer Learning)
########
'''
The below architecture follows a standard pattern of convolutional layers:
    Conv2D layers with different numbers of filters (32 and 64). The number of filters determines the 
    complexity of the learned features. With 32 filters in the first Conv2D layer the model can capture 
    a variety of simple and low-level features such as edges and textures. In the second Conv2D layer with 64 
    filters the model can learn more complex and high-level features such as shapes and patterns.
    Dropout layers randomly set a fraction of input units to 0 during training, which helps to prevent 
    overfitting. By randomly dropping out units dropout regularization introduces noise to the network 
    and forces the model to learn more robust and generalizable features. A dropout rate of 0.25 and 0.5 is 
    applied after the pooling layers and the fully connected layer respectively.
    MaxPooling2D layers helps to retain the most salient features while discarding some spatial information.
    The softmax activation in the output layer provides the final probability distribution over the 10 galaxy 
    classes.
'''
# Building the layers
modelCNN = Sequential()

modelCNN.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
modelCNN.add(Activation('relu'))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))
modelCNN.add(Dropout(0.25))

modelCNN.add(Conv2D(64, (3, 3), padding='same'))
modelCNN.add(Activation('relu'))
modelCNN.add(MaxPooling2D(pool_size=(2, 2)))
modelCNN.add(Dropout(0.25))

modelCNN.add(Flatten())
modelCNN.add(Dense(512))
modelCNN.add(Activation('relu'))
modelCNN.add(Dropout(0.5))

modelCNN.add(Dense(10, activation="softmax"))

modelCNN.summary()

# Compiling the model
modelCNN.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

'''
By including the EarlyStopping callback in the model fitting process the training will be automatically 
stopped if the validation loss does not improve for the specified number of epochs (in this case 5). 
This helps prevent the model from overfitting by avoiding unnecessary training iterations when it's no 
longer improving its performance on the validation set.
'''
# Setting up early stopping during the training of the LeNet-5 model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

# Training the model
history_CNN = modelCNN.fit(X_train, y_train,    
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=early_stopping)

# Model Evaluation
# Test data
# Calculating and printing the test loss and test accuracy.
test_scores = modelCNN.evaluate(X_test, y_test, verbose=2)
print('Test loss for baseline CNN Model:', test_scores[0])
print('Test accuracy for baseline CNN Model:', test_scores[1])

# Data to be fed through confusion matrix and classification report
pred = modelCNN.predict(X_test)
indexes = tf.argmax(pred, axis=1)
i = tf.cast([], tf.int32)
indexes = tf.gather_nd(indexes, i)

# Creating Confusion Matrix
cm = confusion_matrix(y_test, indexes)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
c = ConfusionMatrixDisplay(cm, display_labels=range(10))
c.plot(ax = ax)
ax.set_title('Confusion Matrix (test) for baseline CNN Model', fontsize=16)
plt.show()

# Printing Classification Report
print("Classification report (test) for baseline CNN Model:\n", classification_report(y_test, indexes))

# Training data
# Calculating and printing the train loss and train accuracy.
train_scores = modelCNN.evaluate(X_train, y_train, verbose=2)
print('Train loss for baseline CNN Model:', train_scores[0])
print('Train accuracy for baseline CNN Model:', train_scores[1])

# Data to be fed through confusion matrix and classification report
pred = modelCNN.predict(X_train)
indexes = tf.argmax(pred, axis=1)
i = tf.cast([], tf.int32)
indexes = tf.gather_nd(indexes, i)

# Creating Confusion Matrix
cm = confusion_matrix(y_train, indexes)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
c = ConfusionMatrixDisplay(cm, display_labels=range(10))
c.plot(ax = ax)
ax.set_title('Confusion Matrix (train) for baseline CNN Model', fontsize=16)
plt.show()

# Printing Classification Report
print("Classification report (train) for baseline CNN Model:\n", classification_report(y_train, indexes))

'''
We can evaluate the presence of overfitting by analyzing the accuracy vs. iterations plot and 
the validation loss vs. iterations plot. 
Signs of overfitting can be observed when there is a significant difference between the training and 
validation performance metrics. Specifically if the training accuracy is consistently higher than the 
validation accuracy or if the validation loss is consistently higher or starts to increase while the 
training loss decreases. These discrepancies suggest that the model is likely overfitting the training data 
and not generalizing well to new data.
'''
# Checking for overfitting using Training data
# Accuracy vs. Iterations Plot
plt.plot(history_CNN.history['accuracy'], label='train')
plt.plot(history_CNN.history['val_accuracy'], label='val')
plt.ylabel('accuracy', fontsize=12)
plt.xlabel('iterations', fontsize=12)
plt.title('Accuracy vs. Iterations Plot for baseline CNN Model', fontsize=16)
plt.legend()
plt.show()

# Validation Loss vs. Iterations Plot
plt.plot(history_CNN.history['loss'], label='train')
plt.plot(history_CNN.history['val_loss'], label='val')
plt.ylabel('loss', fontsize=12)
plt.xlabel('iterations', fontsize=12)
plt.title('Loss vs. Iterations Plot for baseline CNN Model', fontsize=16)
plt.legend()
plt.show()

########
# LeNet5 Model
########
'''
The below is LeNet-5 architecture proposed by Yann LeCun et al. in 1998 was primarily trained on the MNIST 
dataset.The MNIST dataset consists of grayscale handwritten digit images with 60,000 training samples and 
10,000 test samples each of size 28x28 pixels. 
The LeNet-5 architecture utilizes:
    Convolutional layers (6 filters in the first Conv2D layer and 16 filters in the second) to extract
    meaningful features from the input images.
    Pooling layers (pool size of (2, 2) and a stride of (2, 2)) to downsample and retain important 
    information.
    Dense layers (1st dense layer has 120 units, 2nd dense layer has 84 units, final dense layer has 10 units)
    to learn complex patterns and make class predictions.
    The combination of these layers enables the model to effectively classify galaxies based on the learned 
    representations.
    The last dense layer with 10 units represents the output layer of the network. It uses the softmax activation 
    function to produce a probability distribution over the 10 possible classes.
'''
model_LeNet5 = Sequential()

# LeNet-5 conv-net architecture
model_LeNet5.add(Conv2D(filters = 6, kernel_size = (5, 5), strides = (1, 1), activation = 'tanh', input_shape= X_train.shape[1:]))
model_LeNet5.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))
model_LeNet5.add(Conv2D(filters = 16, kernel_size = (5, 5), strides = (1, 1), activation = 'tanh'))
model_LeNet5.add(AveragePooling2D(pool_size = (2, 2), strides = (2, 2)))

model_LeNet5.add(Flatten())
model_LeNet5.add(Dense(units = 120, activation = 'tanh'))
model_LeNet5.add(Dense(units = 84, activation = 'tanh'))
model_LeNet5.add(Dense(units = 10, activation = 'softmax'))

model_LeNet5.summary()

# Setting up early stopping during the training of the LeNet-5 model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

# Compiling the model
model_LeNet5.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

# Training the model
history_LeNet5 = model_LeNet5.fit(X_train, y_train,    
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=early_stopping)

# Model Evaluation
# Test data
# Calculating and printing the test loss and test accuracy.
test_scores = model_LeNet5.evaluate(X_test, y_test, verbose=2)
print('Test loss for LeNet5 Model:', test_scores[0])
print('Test accuracy for LeNet5 Model:', test_scores[1])

# Data to be fed through confusion matrix and classification report
pred = model_LeNet5.predict(X_test)
indexes = tf.argmax(pred, axis=1)
i = tf.cast([], tf.int32)
indexes = tf.gather_nd(indexes, i)

# Creating Confusion Matrix
cm = confusion_matrix(y_test, indexes)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
c = ConfusionMatrixDisplay(cm, display_labels=range(10))
c.plot(ax = ax)
ax.set_title('Confusion Matrix (test) for LeNet5 Model', fontsize=16)
plt.show()

# Printing Classification Report
print("Classification report (test) LeNet5 Model:\n", classification_report(y_test, indexes))

# Training data
# Calculating and printing the train loss and train accuracy.
train_scores = model_LeNet5.evaluate(X_train, y_train, verbose=2)
print('Train loss for baseline LeNet5 Model:', train_scores[0])
print('Train accuracy for baseline LeNet5 Model:', train_scores[1])

# Data to be fed through confusion matrix and classification report
pred = model_LeNet5.predict(X_train)
indexes = tf.argmax(pred, axis=1)
i = tf.cast([], tf.int32)
indexes = tf.gather_nd(indexes, i)

# Creating Confusion Matrix
cm = confusion_matrix(y_train, indexes)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
c = ConfusionMatrixDisplay(cm, display_labels=range(10))
c.plot(ax = ax)
ax.set_title('Confusion Matrix (train) for LeNet5 Model', fontsize=16)
plt.show()

# Printing Classification Report
print("Classification report (train) LeNet5 Model:\n", classification_report(y_train, indexes))

# Checking for overfitting using Training data
# Accuracy vs. Iterations Plot
plt.plot(history_LeNet5.history['accuracy'], label='train')
plt.plot(history_LeNet5.history['val_accuracy'], label='val')
plt.ylabel('accuracy', fontsize=12)
plt.xlabel('iterations', fontsize=12)
plt.title('Accuracy vs. Iterations Plot LeNet5 Model', fontsize=16)
plt.legend()
plt.show()

# Validation Loss vs. Iterations Plot
plt.plot(history_LeNet5.history['loss'], label='train')
plt.plot(history_LeNet5.history['val_loss'], label='val')
plt.ylabel('validation loss', fontsize=12)
plt.xlabel('iterations', fontsize=12)
plt.title('Loss vs. Iterations Plot LeNet5 Model', fontsize=16)
plt.legend()
plt.show()

########
# VGG16 Model with Transfer Learning
########
# Create the base VGG16 model
VGG16_model = VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
model_VGG16 = Sequential()

'''
Since we want to use the architecture and pre-trained weights of a pre-existing model (VGG16) as a 
starting point or a base for building a new model. By reusing the pre-trained layers we can benefit from 
the learned features and representations without having to train the entire model from scratch. 
This part is commonly known as transfer learning.
'''
# Adding all the layers of a pre-trained VGG16 model to a new model called model_VGG16
for layer in VGG16_model.layers:
  model_VGG16.add(layer)

'''
The purpose of freezing the base model layers in transfer learning is to prevent them from being updated or 
trained during the training process of the transfer learning model. By freezing the base model layers we 
retain the pre-trained weights and feature extraction capabilities of the base model allowing us to focus 
on training only the added layers (in this case the custom Dense layers and the output layer).
'''
# Freezing the weights of the base model layers
for layer in model_VGG16.layers:
  layer.trainable = False

'''
By adding the Flatten() layer to the VGG16 model we ensure that the output of the last pooling layer is 
flattened before feeding it to the subsequent Dense layers. This correctly shapes the input for the 
subsequent layers and enabling the model to learn from the extracted features.

The additional Dense layers are introduced to further adapt the pre-trained model to the specific 
classification task of galaxies.
By adding a dense layer with 512 units we are increasing the model's capacity to learn and represent 
complex patterns and relationships in the galaxy images. The larger the number of units the more expressive 
power the layer has to capture intricate features specific to galaxy classification.
The additional dense layers further refine the model's ability to learn and represent more specific features 
and patterns relevant to galaxy images.
By gradually reducing the number of units we introduce a level of dimensionality reduction allowing the 
model to focus on more compact and concise representations.
The activation function "softmax" produces a probability distribution over the classes enabling 
multi-class classification. 
'''
# Adding a Flatten() layer to the VGG16 model when setting it up for transfer learning.
model_VGG16.add(Flatten())
# Add custom Dense layers on top of the base model to perform transfer learning
model_VGG16.add(Dense(512))
model_VGG16.add(Dense(64))
model_VGG16.add(Dense(32))
# Add the output layer with softmax activation. 
# Note the 10 in the output layer is the number of classes or categories in our classification problem
model_VGG16.add(Dense(10, activation="softmax")) 

model_VGG16.summary()

# Compiling the new model
model_VGG16.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

# Setting up early stopping during the training of a VGG16 model with transfer learning
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

# Training the new model
history_VGG16 = model_VGG16.fit(X_train, y_train,    
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=early_stopping)

# Model Evaluation
# Test data
# Calculating and printing the test loss and test accuracy.
test_scores = model_VGG16.evaluate(X_test, y_test, verbose=2)
print('Test loss for VGG16 Model with TL:', test_scores[0])
print('Test accuracy for VGG16 Model with TL:', test_scores[1])

# Data to be fed through confusion matrix and classification report
pred = model_VGG16.predict(X_test)
indexes = tf.argmax(pred, axis=1)
i = tf.cast([], tf.int32)
indexes = tf.gather_nd(indexes, i)

# Creating the Confusion Matrix
cm = confusion_matrix(y_test, indexes)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
c = ConfusionMatrixDisplay(cm, display_labels=range(10))
c.plot(ax = ax)
ax.set_title('Confusion Matrix (test) for VGG16 Model with Transfer Learning', fontsize=16)
plt.show()

# Printing the Classification Report
print("Classification report (test) for VGG16 Model with Transfer Learning:\n", classification_report(y_test, indexes))

# Training data
# Calculating and printing the train loss and train accuracy.
train_scores = model_VGG16.evaluate(X_train, y_train, verbose=2)
print('Train loss for VGG16 Model with TL:', train_scores[0])
print('Train accuracy for VGG16 Model with TL:', train_scores[1])

# Data to be fed through confusion matrix and classification report
pred = model_VGG16.predict(X_train)
indexes = tf.argmax(pred, axis=1)
i = tf.cast([], tf.int32)
indexes = tf.gather_nd(indexes, i)

# Creating the Confusion Matrix
cm = confusion_matrix(y_train, indexes)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
c = ConfusionMatrixDisplay(cm, display_labels=range(10))
c.plot(ax = ax)
ax.set_title('Confusion Matrix (train) for VGG16 Model with Transfer Learning', fontsize=16)
plt.show()

# Printing the Classification Report
print("Classification report (train) for VGG16 Model with Transfer Learning:\n", classification_report(y_train, indexes))


# Checking for overfitting using Training data
# Accuracy vs. Iterations Plot
plt.plot(history_VGG16.history['accuracy'], label='train')
plt.plot(history_VGG16.history['val_accuracy'], label='val')
plt.ylabel('accuracy', fontsize=12)
plt.xlabel('iterations', fontsize=12)
plt.title('Accuracy vs. Iterations Plot for VGG16 Model with TL', fontsize=16)
plt.legend()
plt.show()

# Validation Loss vs. Iterations Plot
plt.plot(history_VGG16.history['loss'], label='train')
plt.plot(history_VGG16.history['val_loss'], label='val')
plt.ylabel('loss', fontsize=12)
plt.xlabel('iterations', fontsize=12)
plt.title('Loss vs. Iterations Plot for VGG16 Model with TL', fontsize=16)
plt.legend()
plt.show()

########
# VGG19 Model with Transfer Learning
########
# Create the base VGG19 model
VGG19_model = VGG19(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
model_VGG19 = Sequential()

# Adding all the layers of a pre-trained VGG19 model to a new model called model_VGG19
for layer in VGG19_model.layers:
  model_VGG19.add(layer)

# Freezing the weights of the base model layers
for layer in model_VGG19.layers:
  layer.trainable = False

'''
These additional dense layers increase the model's capacity to learn complex relationships and features 
from the galaxy images. 
Adding a dense layer with 1024 units increases the model's capacity to capture complex patterns.
This larger number of units allows the layer to learn more intricate and fine-grained representations of 
the image features potentially improving the model's ability to discriminate between different types of 
galaxies.
The smaller numbers of units continue the process of dimensionality reduction and abstraction. 
They help the model focus on more compact and higher-level representations of the galaxy images
The activation function "softmax" produces a probability distribution over the classes enabling 
multi-class classification. 
'''
# Adding a Flatten() layer to the VGG16 model when setting it up for transfer learning.
model_VGG19.add(Flatten())
# Add custom Dense layers on top of the base model to perform transfer learning
model_VGG19.add(Dense(1024))
model_VGG19.add(Dense(512))
model_VGG19.add(Dense(64))
model_VGG19.add(Dense(32))
# Add the output layer with softmax activation. 
# Note the 10 in the output layer is the number of classes or categories in our classification problem
model_VGG19.add(Dense(10, activation="softmax"))

model_VGG19.summary()

# Compiling the new model
model_VGG19.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

# Setting up early stopping during the training of a VGG16 model with transfer learning
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

# Training the new model
history_VGG19 = model_VGG19.fit(X_train, y_train,    
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=early_stopping)

# Model Evaluation
# Test data
# Calculating and printing the test loss and test accuracy.
test_scores = model_VGG19.evaluate(X_test, y_test, verbose=2)
print('Test loss for VGG19 Model with TL:', test_scores[0])
print('Test accuracy for VGG19 Model with TL:', test_scores[1])

# Data to be fed through confusion matrix and classification report
pred = model_VGG19.predict(X_test)
indexes = tf.argmax(pred, axis=1)
i = tf.cast([], tf.int32)
indexes = tf.gather_nd(indexes, i)

# Creating the Confusion Matrix
cm = confusion_matrix(y_test, indexes)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
c = ConfusionMatrixDisplay(cm, display_labels=range(10))
c.plot(ax = ax)
ax.set_title('Confusion Matrix (test) for VGG19 Model with Transfer Learning', fontsize=16)
plt.show()

# Printing the Classification Report
print("Classification report (test) for VGG19 Model with Transfer Learning:\n", classification_report(y_test, indexes))

# Training data
# Calculating and printing the train loss and train accuracy.
train_scores = model_VGG19.evaluate(X_train, y_train, verbose=2)
print('Train loss for VGG19 Model with TL:', train_scores[0])
print('Train accuracy for VGG19 Model with TL:', train_scores[1])

# Data to be fed through confusion matrix and classification report
pred = model_VGG19.predict(X_train)
indexes = tf.argmax(pred, axis=1)
i = tf.cast([], tf.int32)
indexes = tf.gather_nd(indexes, i)

# Creating the Confusion Matrix
cm = confusion_matrix(y_train, indexes)
fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
c = ConfusionMatrixDisplay(cm, display_labels=range(10))
c.plot(ax = ax)
ax.set_title('Confusion Matrix (train) for VGG19 Model with Transfer Learning', fontsize=16)
plt.show()

# Printing the Classification Report
print("Classification report (train) for VGG19 Model with Transfer Learning:\n", classification_report(y_train, indexes))


# Checking for overfitting using Training data
# Accuracy vs. Iterations Plot
plt.plot(history_VGG19.history['accuracy'], label='train')
plt.plot(history_VGG19.history['val_accuracy'], label='val')
plt.ylabel('accuracy', fontsize=12)
plt.xlabel('iterations', fontsize=12)
plt.title('Accuracy vs. Iterations Plot for VGG19 Model with TL', fontsize=16)
plt.legend()
plt.show()

# Validation Loss vs. Iterations Plot
plt.plot(history_VGG19.history['loss'], label='train')
plt.plot(history_VGG19.history['val_loss'], label='val')
plt.ylabel('validation loss', fontsize=12)
plt.xlabel('iterations', fontsize=12)
plt.title('Loss vs. Iterations Plot for VGG19 Model with TL', fontsize=16)
plt.legend()
plt.show()