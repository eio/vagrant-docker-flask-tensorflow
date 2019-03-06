# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# print('Tensorflow version:', tf.__version__)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("<{}> {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(True)
  plt.xticks(np.arange(10), class_names, rotation='vertical')
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  # ensure text is not cut off
  plt.tight_layout()
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Returns a short sequential model
def create_model():
    # setup the layers
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # compile the model
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

#######################
#######################
## Retrieve the Data ##
#######################
#######################

# import the MNIST dataset
mnist = keras.datasets.mnist
# specify classification values
class_names = ['0','1','2','3','4','5','6','7','8','9']
# labels are an array of integers, 0 to 9, corresponding with `class_names`
# images are 28x28 NumPy arrays
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

######################
######################
## Explore the Data ##
######################
######################

# print('Train set:')
# print(train_images.shape)
# print(len(train_labels))
# print(train_labels)
# # ^ training set has 60,000 images
# # and 60,000 corresponding labels
# print('Test set:')
# print(test_images.shape)
# print(len(test_labels))
# # ^ training set has 10,000 images
# # and 10,000 corresponding labels

# # display a training image
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(True)
# plt.show()

########################
########################
## Normalize the Data ##
########################
########################

# scale pixel values so that
# instead of 0-255 they are 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

#################################
#################################
## Visualize the Training Data ##
#################################
#################################

# display the first 25 images from the training set
# and display the class name below each image
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

###############################
###############################
## Setup checkpoint callback ##
###############################
###############################

# checkpoint_path = "training_1/cpo.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create checkpoint callback
# cp_callback = keras.callbacks.ModelCheckpoint(
#     checkpoint_path, 
#     save_weights_only=True,
#     verbose=1,
#     # Save weights, every 5 epochs.
#     period=5
# )

##############################
##############################
## Train and save the Model ##
##############################
##############################

# create basic model instance
model = create_model()
# model.save_weights(checkpoint_path.format(epoch=0))
# # train the model, saving with our checkpoint callback
# model.fit(train_images, train_labels, epochs=10, callbacks=[cp_callback])
# train the model
model.fit(train_images, train_labels, epochs=10)
# Save entire model to a HDF5 file
model.save('models/mnist_model.h5')

# evaluate accuracy
loss, acc = model.evaluate(test_images, test_labels)
print("Newly trained model, accuracy: {:5.2f}%".format(100*acc))

######################
######################
## Load saved model ##
######################
######################

# Recreate the exact same model, including weights and optimizer.
model = keras.models.load_model('models/mnist_model.h5')
# model.summary()

# evaluate accuracy of loaded model
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

######################
######################
## Make Predictions ##
######################
######################

# predictions = model.predict(test_images)

#######################
#######################
## Check Predictions ##
#######################
#######################

# # grab the first prediction
# first_prediction = predictions[0]
# # check which of the 10 label integers
# # has the highest probability value
# predicted = np.argmax(first_prediction)

# # check guess against test label
# actual = test_labels[0]
# print
# if actual == predicted:
#     print('Success! First prediction is correct!')
# else:
#     print('Whoops... First prediction failed.')
# # display the 0th image, predictions, and prediction array
# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, test_labels)
# plt.show()

#########################
#########################
## Display Predictions ##
#########################
#########################

# # Plot the first X test images, their predicted label, and the true label
# # Color correct predictions in blue, incorrect predictions in red
# num_rows = 8
# num_cols = 5
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions, test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions, test_labels)
# plt.show()

#######################
#######################
## Single Prediction ##
#######################
#######################

# # Grab an image from the test dataset
# img = np.array([
#   0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,192,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,52,0,180,0,7,0,241,0,0,0,254,0,0,0,255,0,0,0,255,0,3,0,250,0,58,0,176,0,60,0,174,0,128,0,128,0,128,0,191,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,0,0,0,255,0,0,0,205,0,0,0,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,53,0,0,0,205,0,0,0,227,0,0,0,255,0,0,0,255,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,61,0,172,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,94,0,0,0,243,0,0,0,255,0,0,0,56,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,52,0,0,0,254,0,0,0,238,0,0,0,254,0,0,0,255,0,0,0,255,0,0,0,123,0,0,0,255,0,0,0,255,0,0,0,255,0,0,0,238,0,0,0,118,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,0,0,0,255,0,0,0,225,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,255,0,0,0,215,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,0,0,255,0,0,0,254,0,0,0,251,0,0,0,225,0,0,0,239,0,0,0,236,0,0,0,197,0,0,0,191,0,0,0,191,0,0,0,190,0,0,0,182,0,0,0,171,0,0,0,254,0,0,0,255,0,0,0,199,0,0,0,225,0,0,0,246,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,1,0,0,0,44,0,0,0,254,0,0,0,46,0,0,0,75,0,0,0,88,0,0,0,88,0,0,0,85,0,0,0,103,0,0,0,100,0,0,0,98,0,0,0,137,0,0,0,78,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,232,0,0,0,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,252,0,0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,167,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,0,0,0,226,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,87,0,0,0,137,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,0,0,0,254,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,207,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,125,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,245,0,0,0,254,0,0,0,152,0,0,0,0,0,0,0,0,0,0,0,227,0,0,0,37,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,129,0,127,0,128,0,191,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,50,0,182,0,0,0,254,0,0,0,255,0,88,0,151,0,129,0,127,0,129,0,127,0,129,0,127,0,129,0,127,0,128,0,191,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# ])

# # scale it
# img = img / 255

# # Grab an image from the test dataset
# img = test_images[11]
# print(img.shape)

# # Add the image to a batch where it's the only member.
# img_batch = (np.expand_dims(img,0))
# # print(img.shape)

# # Make prediction
# predictions_single = model.predict(img_batch)
# # print(predictions_single)

# prediction = predictions_single[0]
# predicted = np.argmax(prediction)

# # display test image and prediction
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.xlabel("Prediction: {}".format(class_names[predicted]))
# plt.show()
