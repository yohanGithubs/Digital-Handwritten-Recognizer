
# Handwritten digit recognition for MNIST dataset using Convolutional Neural Networks

# Step 1: Import all required keras libraries
import tensorflow as tf
import keras
from tensorflow.keras.utils import to_categorical  # This is used to load mnist dataset later
from tensorflow.keras.datasets import mnist # This will be used to convert your test image to a categorical class (digit from 0 to 9)
from keras.models import load_model 

# Step 2: Load and return training and test datasets
def load_dataset():
	# 2a. Load dataset X_train, X_test, y_train, y_test via imported keras library
	(X_train, y_train), (X_test, y_test) = mnist.load_data()	
	# 2b. reshape for X train and test vars - Hint: X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
	# 2c. normalize inputs from 0-255 to 0-1 - Hint: X_train = X_train / 255
	X_train = X_train / 255
	X_test = X_test / 255
	# 2d. Convert y_train and y_test to categorical classes - Hint: y_train = np_utils.to_categorical(y_train)
	y_train = to_categorical(y_train)
	y_test = to_categorical(y_test)
	# 2e. return your X_train, X_test, y_train, y_test
	return X_train, X_test, y_train, y_test
# Step 3: Load your saved model 
cnnModel = load_model('digitRecognizer')
# Step 4: Evaluate your model via your_model_name.evaluate(X_test, y_test, verbose = 0) function

X_train, X_test, y_train, y_test = load_dataset()
modelScores = cnnModel.evaluate(X_test, y_test, verbose = 0)
print("Accuracy for dataset is: %.2f%%" % ( modelScores[1] * 100))

# Code below to make a prediction for a new image.

# Step 5: This section below is optional and can be copied from your digitRecognizer.py file from Step 8 onwards - load required keras libraries
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model
# Step 6: load and normalize new image
def load_new_image(path):
	# 9a. load new image
	newImage = tf.keras.preprocessing.image.load_img(path, color_mode='grayscale', target_size=(28, 28))
	# 9b. Convert image to array
	newImage = tf.keras.preprocessing.image.img_to_array(newImage)
	# 9c. reshape into a single sample with 1 channel (similar to how you reshaped in load_dataset function)
	newImage = newImage.reshape(1, 28, 28, 1)
	# 9d. normalize image data - Hint: newImage = newImage / 255
	newImage = newImage / 255
	# 9e. return newImage
	return newImage

# Step 7: load a new image and predict its class
def test_model_performance(model):
	# 10a. Call the above load image function
	digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	
	img_paths = [f'sample_images/digit{x}.png' for x in range(1, 10)]
	images = np.array([load_new_image(img_path) for img_path in img_paths]) # resize and reshape to fit model input
	images = np.vstack(images) #stack the list of immage arrays into a single array

		# 10d. Print prediction result
	predictions = model.predict(images)
	#iterate over predictions to print the predicted digit
	for i, prediction in enumerate(predictions):
			predicted_digit = digits[np.argmax(prediction)] # get the index of the max value in the prediction array
			print(f"Digit {i+1}: Predicted digit is {predicted_digit}") # print the index i + 1 caue index starts from 0

# Step 11: Test model performance here by calling the above test_model_performance function
model = load_model('digitRecognizer')
test_model_performance(model)
