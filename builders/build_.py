# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from helpers.io import HDF5DatasetWriter_face
from helpers.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import random
import cv2
import os
import sys

class Build:
	def __init__(self, config):
		# intialize the configuration
		self.config = config
		self.rows = list(paths.list_images(self.config.image_Path))[:5000]

	def build(self):
		# obtain the train, validation, and testing images
		# partition data into training, testing, and validation splits
		print("[INFO] constructing the training data...")
		split = train_test_split(self.rows, targets, test_size = 0.2,
			random_state=42)
		(trainPaths, testPaths, trainLabels, testLabels) = split

		print("[INFO] constructing the testing and validation data...")
		split = train_test_split(testPaths, testLabels, test_size = 0.1,
			random_state=42)
		(valPaths, testPaths, valLabels, testLabels) = split

		# construct a list pairing training, validation and testing
		datasets = [
		("train", trainPaths, trainLabels, self.config.train_hdf5),
		("test", testPaths, testLabels, self.config.test_hdf5),
		("val", valPaths, valLabels, self.config.val_hdf5)]

		# construct a list pairing training, validation and testing
		datasets = [
		("train", trainPaths, trainLabels, self.config.train_hdf5),
		("test", testPaths, testLabels, self.config.test_hdf5),
		("val", valPaths, valLabels, self.config.val_hdf5)]

		# initialize the list for averages of the RGB channel
		(R, G, B) = ([], [], [])

		# loop over the dataset tuples
		for (dType, paths, labels, outputPath) in datasets:
			# create the HDF5 writer
			print("[INFO] building {} set...".format(dType))
			writer = HDF5DatasetWriter((len(paths), 224, 224, 3), outputPath)

			# initialize the progress bar
			widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
				progressbar.Bar(), " ", progressbar.ETA()]
			pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

			# loop over the image paths
			for (i, (path, label)) in enumerate(zip(paths, labels)):
				# load the image path and process it
				image = cv2.imread(path)

				# if we are building the training then compute mean of each channel in
				# the image, update this list
				if dType == "train":
					(b, g, r) = cv2.mean(image)[:3]
					R.append(r)
					G.append(g)
					B.append(b)

				# add image and label to the HDF5 dataset
				writer.add([image], [label])
				# update progress bar
				pbar.update(i)
				
			# close the hdf5 writer
			pbar.finish()
			writer.close()

		# serializing the means
		print("[INFO] Serializing means...")
		D = {"R":np.mean(R)/255, "G":np.mean(G)/255, "B":np.mean(B)/255}
		f = open(self.config.color_mean, "w")
		f.write(json.dumps(D))
		f.close()


	# build the database
	def face_build(self):
		# grab the path to images and initialize targets
		targets = []

		# generate the labels
		for (count, row) in enumerate(self.rows):
			print("[INFO] Generating labels {} of {}...".format(count + 1, len(self.rows)))
			
			# As images have been resized to (224, 224) randomly generate four numbers
			# to add some randomness to the generated bounding box
			startx = random.uniform(0, 0.15)
			starty = random.uniform(0, 0.15)
			endx = random.uniform(0.85, 1)
			endy = random.uniform(0.85, 1)

			# update list of data, targets, and filenames
			targets.append((startx, starty, endx, endy))

		# obtain the train, validation, and testing images
		# partition data into training, testing, and validation splits
		print("[INFO] constructing the training data...")
		split = train_test_split(self.rows, targets, test_size = 0.2,
			random_state=42)
		(trainPaths, testPaths, trainLabels, testLabels) = split

		print("[INFO] constructing the testing and validation data...")
		split = train_test_split(testPaths, testLabels, test_size = 0.1,
			random_state=42)
		(valPaths, testPaths, valLabels, testLabels) = split

		# construct a list pairing training, validation and testing
		datasets = [
		("train", trainPaths, trainLabels, self.config.train_hdf5),
		("test", testPaths, testLabels, self.config.test_hdf5),
		("val", valPaths, valLabels, self.config.val_hdf5)]

		# initialize the list for averages of the RGB channel
		(R, G, B) = ([], [], [])

		# loop over the dataset tuples
		for (dType, paths, labels, outputPath) in datasets:
			# create the HDF5 writer
			print("[INFO] building {} set...".format(dType))
			writer = HDF5DatasetWriter_face((len(paths), 224, 224, 3), outputPath)

			# initialize the progress bar
			widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
				progressbar.Bar(), " ", progressbar.ETA()]
			pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

			# loop over the image paths
			for (i, (path, label)) in enumerate(zip(paths, labels)):
				# load the image path and process it
				image = cv2.imread(path)

				# image conversion rescaling
				image = np.array(image, dtype="float32") / 255
				label = np.array(label, dtype="float32")
				
				# if we are building the training then compute mean of each channel in
				# the image, update this list
				if dType == "train":
					(b, g, r) = cv2.mean(image)[:3]
					R.append(r)
					G.append(g)
					B.append(b)

				# add image and label to the HDF5 dataset
				writer.add([image], [label])
				# update progress bar
				pbar.update(i)

			# close the hdf5 writer
			pbar.finish()
			writer.close()

		# serializing the means
		print("[INFO] Serializing means...")
		D = {"R":np.mean(R)/255, "G":np.mean(G)/255, "B":np.mean(B)/255}
		f = open(self.config.color_mean, "w")
		f.write(json.dumps(D))
		f.close()

# End of code