# Used fine tuning to train a model that predicts the bounding box
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from helpers.learningratefinder import LearningRateFinder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from helpers.clr_callback import CyclicLR
from Config import config_bbox as config
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import sys
import os

# construct the argument parser 
ap = argparse.ArgumentParser()
ap.add_argument('-ans', "--answer", required=True,
	help="Has the training, validation, testing set been created? [Y/N]")
ap.add_argument("-f", "--lr-find", type=int, default=0,
	help="whether or not to find optimal learning rate")
args = vars(ap.parse_args())

# setting up the policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

if args["answer"].upper() == "N" or args["answer"].upper() == "No":
	# grab database of processed images
	print("[INFO] Loading the face annotations and making dataset split...")
	rows = list(paths.list_images(config.image_Path))

	# initialize the list of images, target output predictions, individual file names
	filenames = []
	targets = []
	data = []

	# loop through all image paths
	for (count, row) in enumerate(rows):
		print("[INFO] Processing image {} of {}...".format(count + 1, len(rows)))
		row_split = row.split("\\")
		filename = row_split[-1]

		# read image
		image = cv2.imread(row)
		[h,w] = image.shape[:2]

		# As images have been resized to (224, 224) randomly generate four numbers
		# to add some randomness to the generated bounding box
		start_coeff_x = random.uniform(0, 0.2)
		start_coeff_y = random.uniform(0, 0.2)
		end_coeff_x = random.uniform(0.8, 1)
		end_coeff_y = random.uniform(0.8, 1)


		# randomly obtain and scale the points
		startx = int(start_coeff_x * w) / w
		starty = int(start_coeff_y * h) / h
		endx = int(end_coeff_x * w) / w
		endy = int(end_coeff_y * h) / h

		# preprocess the image
		image = img_to_array(image)

		# update list of data, targets, and filenames
		targets.append((startx, starty, endx, endy))
		filenames.append(filename)
		data.append(image)

	# convert the data and targets to numpy arrays -> rescaling the inputs
	data = np.array(data, dtype="float32") / 255.0
	targets = np.array(targets, dtype="float32")

	# partition data into training, testing, and validation splits
	print("[INFO] constructing the training data...")
	split = train_test_split(data, targets, filenames, test_size = 0.2,
		random_state=42)
	(trainImages, testImages, trainTargets, testTargets, trainFilenames, testFilenames) = split

	print("[INFO] constructing the testing and validation data...")
	split = train_test_split(testImages, testTargets, testFilenames, test_size = 0.1,
		random_state=42)
	(valImages, testImages, valTargets, testTargets, valFilenames, testFilenames) = split

	# store to disk
	print("[INFO] Storing train data...")
	f = open(config.train_data, "w")
	f.write("\n".join(trainImages))
	f.close()

	print("[INFO] Storing train names...")
	f = open(config.train_name, "w")
	f.write("\n".join(trainFilenames))
	f.close()

	print("[INFO] Storing train targets...")
	f = open(config.train_targ, "w")
	f.write("\n".join(trainTargets))
	f.close()

	# store to disk
	print("[INFO] Storing test data...")
	f = open(config.test_data, "w")
	f.write("\n".join(testImages))
	f.close()

	print("[INFO] Storing test names...")
	f = open(config.test_name, "w")
	f.write("\n".join(testFilenames))
	f.close()

	print("[INFO] Storing test targets...")
	f = open(config.test_targ, "w")
	f.write("\n".join(testTargets))
	f.close()

	# store to disk
	print("[INFO] Storing val data...")
	f = open(config.val_data, "w")
	f.write("\n".join(valImages))
	f.close()

	print("[INFO] Storing val names...")
	f = open(config.val_name, "w")
	f.write("\n".join(valFilenames))
	f.close()

	print("[INFO] Storing val targets...")
	f = open(config.val_targ, "w")
	f.write("\n".join(valTargets))
	f.close()

else:
	# else go straight to the dataset
	trainImages = open(config.train_data).read().strip().split("\n")
	trainTargets = open(config.train_trag).read().strip().split("\n")
	valImages = open(config.val_data).read().strip().split("\n")
	valTargets = open(config.val_trag).read().strip().split("\n")

# construct the image generator for data augmentation
aug = ImageDataGenerator(horizontal_flip=True, fill_mode="nearest")
# Fine tuning process begins
# load the VGG16 network, without the top
vgg = VGG16(weights="imagenet", include_top=False,
		input_tensor=Input(config.shape))

# freeze layers to prevent updating during training
vgg.trainable = False

# flatten to max pool of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# create the fully connected head to output the predicted coorindates
boxHead = Dense(128, activation="relu")(flatten)
boxHead = Dense(64, activation="relu")(boxHead)
boxHead = Dense(32, activation="relu")(boxHead)

# final layer that performs the prediction
boxHead = Dense(4, dtype='float32', activation="sigmoid")(boxHead)

# combine all
model = Model(inputs=vgg.input, outputs=boxHead)

# initialize the optimer, compile the model, and show the model -> tweak here
print("[INFO] Compiling model....")
opt = SGD(lr=config.min_lr, momentum=0.9)
model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
print(model.summary())

# check to see if we are attempting to find an optimal learning rate
if args["lr_find"] > 0:
	# initialize the learning rate finder and train with leaning rates from
	# 1e-10 to 1e+1
	print("[INFO] Finding learning rate...")
	lrf = LearningRateFinder(model)
	lrf.find(
		aug.flow(trainImages, trainTargets, batch_size=config.batch_size),
		1e-10, 1e+1,
		stepsPerEpoch=np.ceil(len(trainImages)/ float(config.batch_size)),
		batchSize=config.batch_size,
		epochs=5)

	# plot the loss
	lrf.plot_loss()
	plt.savefig(config.lrfind_path)

	# exit the ting
	print("[INFO] learning rate finder complete")
	print("[INFO] examine plot and adjust learning rates before training")
	sys.exit(0)

# using defined learning rates
stepsize = config.step_size * (trainImages.shape[0] // config.batch_size)
clr = CyclicLR(
	mode=config.CLR_METHOD,
	base_lr=config.MIN_LR,
	max_lr=config.MAX_LR,
	step_size=stepSize)

# train the network for bounding box regression
print("[INFO] Training bounding box regrssor...")
H = model.fit(
	x=aug.flow(trainImages, trainTargets, batch_size=config.batch_size),
	validation_data=(valImages, valTargets),
	steps_per_epoch=trainImages.shape[0] // config.batch_size,
	epochs=config.num_epochs,
	callbacks=[clr],
	verbose=1)

# serlaize the model to disk
print("[INFO] Saving object model...")
model.save(config.output_Path, save_format="h5")

# plot the model
N = config.num_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding box regression loss on training set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.training_path) # tweak this also

# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.clr_plot_path)


	
