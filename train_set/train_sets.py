# Used fine tuning to train a model that predicts the bounding box
from helpers.io import HDF5DatasetGenerator_face
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten
from helpers.io import HDF5DatasetGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from helpers.nn import AutoNet

# obtaining classes 
def class_length(config, dataset):
	classes = 2
	
	# age dataset
	if dataset.lower() == "age":
		classes = len(config.bin_age)

	# sex dataset
	elif dataset.lower() == "sex":
		classes = len(config.sex)

	# race
	elif dataset.lower() == "race":
		classes = len(config.race)

	elif dataset.lower() == "face":
		# predicting 4 values
		classes = 4

	else:
		classes = len(config.emotions)

	return classes

# compiling the program
def compile_trained(config, mp, dataset, aug=None, valAug=None):

	# obtain the class length
	classes = class_length(config, dataset)

	# initialize the generators and loss type
	trainGen = []
	valGen = []
	loss = "categorical_crossentropy"

	if dataset.lower() == "face":
		# initialize the training and validation dataset gens
		trainGen = HDF5DatasetGenerator_face(config.train_hdf5, config.batch_size,
		 aug=aug, preprocessors=[mp], binarize=False)
		valGen = HDF5DatasetGenerator_face(config.val_hdf5, config.batch_size,
		 preprocessors=[mp], binarize=False, aug=valAug)

		# loss type
		loss = "mse"

	else:
		# initialize the training and validation dataset gens
		trainGen = HDF5DatasetGenerator(config.train_hdf5, config.batch_size,
			aug=aug, preprocessors=[mp], classes=classes)
		valGen = HDF5DatasetGenerator(config.val_hdf5, config.batch_size,
			preprocessors=[mp], classes=classes, aug=valAug)

	# also if number of class is two then binary crossentropy
	if classes == 2:
		loss = "binary_crossentropy"

	# build the model
	print("[INFO] Compiling model")
	model = AutoNet.build(config.shape, classes, dataset, reg=1e-5)
	opt = SGD(lr=config.min_lr, momentum=0.9)
	model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

	# return the model and generators
	return model, trainGen, valGen
