# matplot lib backend
import matplotlib
matplotlib.use("Agg")

# Used fine tuning to train a model that predicts the bounding box
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helpers.callbacks.learningratefinder import LearningRateFinder
from train_set import compile_pretrained, compile_trained
from tensorflow.keras.callbacks import ModelCheckpoint
from helpers.callbacks.clr_callback import CyclicLR
from helpers.meanpre import MeanPreprocessor
from tensorflow.keras.optimizers import SGD
from helpers.io import HDF5DatasetGenerator
from helpers.plots import TrainingMonitor
from config import PathConfig
from builders import Build
import numpy as np
import argparse
import json
import cv2
import sys
import os

# construct the argument parser 
ap = argparse.ArgumentParser()
ap.add_argument('-dd', "--dataset", required=True,
	help="Name of dataset")
ap.add_argument('-ans', "--answer", required=True,
	help="Has the training, validation, testing set been created? [Y/N]")
ap.add_argument("-f", "--lr-find", type=int, default=0,
	help="whether or not to find optimal learning rate")
ap.add_argument("-ss", "--startPoint", type=int, default=0,
	help="where to continue training from")
args = vars(ap.parse_args())

# initialize dataset configuration
config = PathConfig(args["dataset"].lower())

# check if we are to build the database
if args["answer"].upper() == "N" or args["answer"].upper() == "No":
	# grab database of processed images
	print("[INFO] Building sets for {} data...".format(args["dataset"]))
	
	# initialize the database
	create_ = Build(config)
	
	# build database
	if args["dataset"].lower() == "face":
		create_.face_build()
	else:
		create_.build()

# initialize the database mean
means = json.loads(open(config.color_mean).read())
mp = MeanPreprocessor(means["R"], means["G"], means["B"])

# image augmentation 
aug = ImageDataGenerator(horizontal_flip=True, fill_mode="nearest")

# construct the image generator for data augmentation
if args["dataset"].lower() == "face":

	# initialize the training and validation dataset gens
	(model, trainGen, valGen) = compile_trained(config, mp, aug=aug)	

# for other datasets	
else:
	
	# initialize the training and validation dataset gens
	trainGen = HDF5DatasetGenerator(config.train_hdf5, config.batch_size,
		  aug=aug)
	valGen = HDF5DatasetGenerator(config.val_hdf5, config.batch_size)
 

# check to see if we are attempting to find an optimal learning rate
if args["lr_find"] > 0:
	# initialize the learning rate finder and train with leaning rates from
	# 1e-10 to 1e+1
	print("[INFO] Finding learning rate...")
	lrf = LearningRateFinder(model)
	lrf.find(trainGen.generator(),
		1e-10, 1e+1,
		stepsPerEpoch=np.ceil(trainGen.numImages / float(config.batch_size)),
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
stepSize = config.step_size * (trainGen.numImages // config.batch_size)
clr = CyclicLR(
	mode=config.clr_method,
	base_lr=config.min_lr,
	max_lr=config.max_lr,
	step_size=stepSize)

# construct path for callbacks
figPath = os.path.sep.join([config.train_loss_path,
 "{}.png".format(os.getpid())])

jsonPath = os.path.sep.join([config.train_loss_path,
 "{}.json".format(os.getpid())])

bname = os.path.sep.join([config.output_Path,
	"weights.hdf5"])

# create call backs
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
	save_best_only=True, verbose=1)
callbacks = [clr, checkpoint, TrainingMonitor(figPath, jsonPath, args["startPoint"])]

# train the network for bounding box regression
print("[INFO] Training bounding box regrssor...")
H = model.fit(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.batch_size,
	validation_data=trainGen.generator(),
	validation_steps=valGen.numImages // config.batch_size,
	epochs=config.num_epochs,
	callbacks=callbacks,
	verbose=1)


# plot the learning rate history
N = np.arange(0, len(clr.history["lr"]))
plt.figure()
plt.plot(N, clr.history["lr"])
plt.title("Cyclical Learning Rate (CLR)")
plt.xlabel("Training Iterations")
plt.ylabel("Learning Rate")
plt.savefig(config.clr_plot_path)

# serlaize the model to disk
print("[INFO] Saving object model...")
model.save(config.save_path, save_format="h5")

# close the dataset
trainGen.close()
valGen.close()


	
