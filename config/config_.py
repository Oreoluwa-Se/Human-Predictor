# import the necessary packages
import numpy as np
import os

# input configuration class -> creates paths for respective datasets
class PathConfig:
	def __init__(self, dataset):
		# modfidy name
		folder_name = dataset[0].upper() + dataset[1:]
		folder = "{}_Detector".format(folder_name)
		
		# initialize parameters and paths
		self.shape = (224, 224, 3)

		# base path
		self.base_Path = "D:\\Artificial_Intelli\\Courses\\PyimageSearch\\Deep_Learning\\Projects\\FD-AGR"

		# image and annotation path
		self.image_Path = os.path.sep.join([self.base_Path, "cropped"])

		# data storage path
		self.data_path = os.path.sep.join([self.base_Path, "Output", folder])
		self.set_storage = os.path.sep.join([self.data_path, "total_{}.txt".format(dataset)])

		# HDF5 output
		self.train_hdf5 = os.path.sep.join([self.data_path, "HDF5", "train.hdf5"])
		self.test_hdf5 = os.path.sep.join([self.data_path, "HDF5", "test.hdf5"])
		self.val_hdf5 = os.path.sep.join([self.data_path, "HDF5", "val.hdf5"])

		# define the output path -> serlaied model, model training plot, testing file names
		self.output_Path = os.path.sep.join([self.data_path, "Model_Output"])

		# define the clr configurations
		self.clr_Path = os.path.sep.join([self.data_path, "Model_Output", "CLR"])
		self.lrfind_path = os.path.sep.join([self.clr_Path, "lrfind_plot.png"])
		self.training_path = os.path.sep.join([self.clr_Path, "training_plot.png"])
		self.clr_plot_path = os.path.sep.join([self.clr_Path, "clr_plot.png"])

		# serialized means
		self.color_mean = os.path.sep.join([self.output_Path, "{}_mean.json".format(dataset)])

		# serialized means
		self.weights = os.path.sep.join([self.output_Path, "{}_weights.json".format(dataset)])

		# model save path
		self.save_path = os.path.sep.join([self.output_Path,"{}.h5".format(dataset)])

		# model save path
		self.train_loss_path = os.path.sep.join([self.output_Path,"CLR", "plots"])

		# change after clr process | For fine tuning
		self.min_lr = 1
		self.max_lr = 1e1
		self.batch_size = 64
		self.step_size = 8
		self.clr_method = "triangular2"
		self.num_epochs = 50

		# age brackets
		self.bin_age = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(21, 24)", "(25, 32)",
			"(33, 37)", "(38, 43)", "(44, 47)", "(48, 53)", "(54, 59)", "(60, 100)",
			 "100+"]

		# race brackets
		self.race = ["White", "Black", "Asian", "Indian", "N/A"]

		# sex
		self.sex = ["m", "f"]

		# emotions
		self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprised", 
			"Neutral", "Uncertain"]

		# fer dataset
		self.fer = "D:\\Artificial_Intelli\\Courses\\PyimageSearch\\Deep_Learning\\Dataset\\Faces\\Images\\fer2013\\fer2013.csv"



		




