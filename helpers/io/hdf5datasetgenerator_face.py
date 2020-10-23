# import necessary packages
from tensorflow.keras.utils import to_categorical
import h5py
import numpy as np

class HDF5DatasetGenerator_face:
	def __init__(self, dbPath, batchSize, preprocessors=None, aug=None,
		binarize=True, classes=2):
		
		# initialize attributes
		self.dbPath = dbPath
		self.batchSize = batchSize
		self.preprocessors = preprocessors
		self.aug = aug
		self. binarize = binarize
		self.classes = classes

		# Open HDF5 data and determine number of entries
		self.db = h5py.File(self.dbPath, "r")
		self.numImages = self.db["labels"].shape[0]

	def generator(self, passes=np.inf):
		# initialze epoch count - in out case we are running
		# until keras reaches training termination criteria.
		# or we stop it outselves.
		epochs = 0

		# Keep looping infiinitely, model stops when weve reached desired
		# epoch count
		while epochs < passes:
			# loop over the HDF5 dataset
			for i in np.arange(0, self.numImages, self.batchSize):
				# extract
				images = self.db["images"][i:i + self.batchSize]
				labels = self.db["labels"][i:i + self.batchSize]

				# chek if labels should be binarized
				if self.binarize:
					labels = to_categorical(labels, self.classes)

				# check to se if preprocessors are not None
				if self.preprocessors is not None:
					# initialize list of processed images
					procImages = []

					# loop over the images
					for image in images:
						# loop over preprocessors
						for p in self.preprocessors:
							image = p.preprocess(image)

						# update the processed list
						procImages.append(image)

					# update the image array to reflect processed images
					images = np.array(procImages)

				# if the data agumento exists - next interates through all options
				if self.aug is not None:
					(images, labels) = next(self.aug.flow(images, labels,
											batch_size=self.batchSize))

				# yield image and labels - we use yield because we are not storing
				# the altered images or the laabels on memory
				yield(images, labels)
			# increment epoch
			epochs +=1

	def close(self):
		# close the database
		self.db.close()


