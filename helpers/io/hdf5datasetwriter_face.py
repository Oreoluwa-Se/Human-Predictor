# import necessary packages
import h5py
import os

class HDF5DatasetWriter_face:
	def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
		# check to see if output path exists, and if so raise exception
		if os.path.exists(outputPath):
			raise ValueError("The supplied path exists and cannot be overwritten "
				"manually delete file before continuing", outputPath)

		# If path doesnt exist open hdf5 database for writing
		# and creating two datasets. one for images/features
		# the other for class labels not encoded for bounding box prediction)

		self.db = h5py.File(outputPath,"w")
		self.data = self.db.create_dataset(dataKey, dims, dtype="float")
		self.labels = self.db.create_dataset("labels", (dims[0], 4), dtype="float")	

		# store the buffer size, and then initialize buffer
		self.bufSize = bufSize
		self.buffer = {"data": [], "labels": []}
		# helps identify begining of empty space
		self.idx = 0 

	# function to add data and labels to the file
	def add(self, rows, labels):
		# add the rows and labels to the buffer
		self.buffer["data"].extend(rows)
		self.buffer["labels"].extend(labels)

		# check to see if buffer is full and we should flush to disk
		if len(self.buffer["data"]) >= self.bufSize:
			self.flush()

	# function to flush to disk
	def flush(self):
		# write buffer to disk and reset
		i = self.idx + len(self.buffer["data"])
		self.data[self.idx:i] = self.buffer["data"]
		self.labels[self.idx:i] = self.buffer["labels"]
		self.idx = i
		self.buffer = {"data": [], "labels": []}
		

	# to close the dataset
	def close(self):
		# check files still in buffer and flush
		if len(self.buffer["data"]) > 0:
			self.flush()

		# close dataset
		self.db.close()





