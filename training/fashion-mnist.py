# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
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
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
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

#######################
#######################
## Retrieve the Data ##
#######################
#######################

# import the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
# specify classification values
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# labels are an array of integers, 0 to 9, corresponding with `class_names`
# images are 28x28 NumPy arrays
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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

#####################
#####################
## Build the Model ##
#####################
#####################

# setup the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc * 100, '%')

######################
######################
## Make Predictions ##
######################
######################

predictions = model.predict(test_images)

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

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 8
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

#######################
#######################
## Single Prediction ##
#######################
#######################

# Grab an image from the test dataset
img = test_images[11]
# print(img.shape)

# Add the image to a batch where it's the only member.
img_batch = (np.expand_dims(img,0))
# print(img.shape)

# Make prediction
predictions_single = model.predict(img_batch)
# print(predictions_single)

prediction = predictions_single[0]
predicted = np.argmax(prediction)

# display test image and prediction
plt.figure()
plt.imshow(img, cmap='gray')
plt.xlabel("Prediction: {}".format(class_names[predicted]))
plt.show()
