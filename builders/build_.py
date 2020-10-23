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
import sys
import os

class Build:
	def __init__(self, config):
		# intialize the configuration
		self.config = config
		self.rows = list(paths.list_images(self.config.image_Path))[:5000]

	def age_index(self, age):
		# loop till we find match
		for i in range(0, len(self.config.bin_age)):
			if self.config.bin_age[i] == age:
				return i

	def race_check(self, race):
		# loop till we find match
		for i in range(0, len(self.config.race)):
			if self.config.bin_age[i] == race:
				return i

	def build(self, dataset):
		# variables
		trainPaths = []
		trainLabels = []

		# generates path and labels for face dataset
		if dataset.lower() == "face":
			trainPaths = self.rows
			# generate the labels
			for (count, row) in enumerate(trainPaths):			
				# As images have been resized to (224, 224) randomly generate four numbers
				# to add some randomness to the generated bounding box
				startx = random.uniform(0, 0.15) * self.config.shape[1]
				starty = random.uniform(0, 0.15) * self.config.shape[1]
				endx = random.uniform(0.85, 1) * self.config.shape[1]
				endy = random.uniform(0.85, 1) * self.config.shape[1]

				# update list of data, targets, and filenames
				trainLabels.append((int(startx), int(starty), int(endx), int(endy)))

		# for other datasets
		else:
			# obtain dataset
			rows = open(self.config.set_storage).read()
			rows = rows.strip().split("\n")[:5000]

			# generate the labels -> redirect to cropped images
			for (i, row) in enumerate(rows):
				# find path to cropped images
				path = row.split("\t")[0]
				path = path.split("\\")[-1]
				path_ = path.rfind(".")

				path = os.path.sep.join([self.config.image_Path, "{}.png".format(path[:path_])])

				# append label and path
				trainPaths.append(path)

				# age dataset
				if dataset.lower() == "age":
					check = Build.age_index(row.split("\t")[-1])

				# sex dataset
				elif dataset.lower() == "sex":
					if row.split("\t")[-1] == "m":
						check = 0
					else:
						check = 1

				# race
				elif dataset.lower == "race":
					check = Build.race_check(row.split("\t")[-1])

				else:
					check = int(row.split("\t")[-1]) - 1

				trainLabels.append(check)

		# add fer to dataset
		count = 1 
		trainPaths = []
		trainLabels = []
		if dataset.lower() == "emotion":
			# open up the fer dataset and skip first line
			rows = open(self.config.fer)
			rows.__next__()

			for row in rows:
				(label, image, usage) = row.strip().split(",")
				label = int(label)

				# we store the image string as the tag
				trainPaths.append(image)
				trainLabels.append(label)
				if count == 5000:
					break
				else:
					count += 1


		# partition data into training, testing, and validation splits
		print("[INFO] constructing the training data...")
		split = train_test_split(trainPaths, trainLabels, test_size = 0.2,
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

		# class weights
		class_w = np.zeros(len(set(trainLabels)))

		# loop over the dataset tuples
		for (dType, paths, labels, outputPath) in datasets:
			# create the HDF5 writer
			print("[INFO] Building {} set...".format(dType))
			(width, height, depth) = self.config.shape

			if dataset.lower() == "face":
				writer = HDF5DatasetWriter_face((len(paths), width, height, depth), outputPath)

			# for other datasets
			else:
				writer = HDF5DatasetWriter((len(paths), width, height, depth), outputPath)

			# initialize the progress bar
			widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
				progressbar.Bar(), " ", progressbar.ETA()]
			pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

			image = []

			# loop over the image paths
			for (i, (path, label)) in enumerate(zip(paths, labels)):

				# if we have no address split then we're working with cifar
				if path.rfind("\\") == -1:
					image = np.array(path.split(" "), dtype="float32")
					image = image.reshape((48, 48))
					image = cv2.merge([image, image, image])

					# scale to 224, 224
					image = cv2.resize(image, (self.config.shape[0], self.config.shape[1]),
					 interpolation=cv2.INTER_CUBIC)

				else:

					# load the image path and process it
					image = cv2.imread(path)
					# load and scale
					image = np.array(image, dtype="float32")

				# convert the targets to float 32 for face set only
				if dataset.lower() == "face":
					label = np.array(label, dtype="float32")

				# obtain mean of channels during the train set
				if dType == "train":
					(b, g, r) = cv2.mean(image)[:3]
					R.append(r)
					G.append(g)
					B.append(b)

					# only store weights for classes that aren't
					if dataset.lower() != "face":
						class_w[label] += 1

				# add image and label to the HDF5 dataset
				writer.add([image], [label])
				# update progress bar
				pbar.update(i)
				
			# close the hdf5 writer
			pbar.finish()
			writer.close()

		# serializing the means
		print("[INFO] Serializing means...")
		D = {"R":np.mean(R), "G":np.mean(G), "B":np.mean(B)}
		f = open(self.config.color_mean, "w")
		f.write(json.dumps(D))
		f.close()

		if dataset.lower() != "face":
			# serialize class weights
			print("[INFO] Storing class weights")
			class_w = class_w.max()/class_w
			D = {"Weights": list(class_w)}
			f = open(self.config.weights, "w")
			f.write(json.dumps(D))
			f.close()

