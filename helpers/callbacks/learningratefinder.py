# import the necessary pacakages
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile

class LearningRateFinder:
	def __init__(self, model, stopFactor=4, beta=0.98):
		# store model, stopfactor, and beta value for computing smoothed average loss
		self.model = model
		self.stopFactor = stopFactor
		self.beta = beta

		# initialize the list of learning rates and losses
		self.lrs = []
		self.losses = []

		# initialize learning rate multiplier, average loss, best loss so far
		# current batch number, and weights file
		self.lrMult = 1
		self.avgLoss = 0
		self.bestLoss = 1e9
		self.batchNum = 0
		self.weightsFile = None

	def reset(self):
		# re-initializes all variables from computer
		self.lrs = []
		self.losses = []
		self.lrMult = 1
		self.avgLoss = 0
		self.bestLoss = 1e9
		self.batchNum = 0
		self.weightsFile = None

	def is_data_iter(self, data):
		# define the set of class types to check for
		iterClasses = ["NumpyArrayIterator", "DirectoryIterator", "DataframeIterator",
		"Sequence", "generator"]

		# return wether the data is an iterator
		return data.__class__.__name__ in iterClasses

	def on_batch_end(self, batch, logs):
		# grab the current learning rate and log it to the list of learning rates
		# we've tried so far
		lr = K.get_value(self.model.optimizer.lr)
		self.lrs.append(lr)

		# grab the loss at the end of the batch, increment the total number of batches
		# compute the average loss, smooth it then update the losses list with the 
		# smoothed value
		l = logs["loss"]
		self.batchNum += 1
		self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
		smooth = self.avgLoss / (1 - (self.beta**self.batchNum))
		self.losses.append(smooth)

		# compute the maximum loss stopping factor value
		stopLoss = self.stopFactor * self.bestLoss

		# check to see whether the loss has grown too large
		if self.batchNum > 1 and smooth > stopLoss:
			# stop returning and leave the method
			self.model.stop_training = True
			return

		# check to see if best loss should be update
		if self.batchNum == 1 or smooth < self.bestLoss:
			self.bestLoss = smooth

		# increase the learning rate
		lr *= self.lrMult
		K.set_value(self.model.optimizer.lr, lr)

	def find(self, trainData, startLR, endLR, epochs=None, stepsPerEpoch=None,
		batchSize=32, sampleSize=2048, verbose=1):
		# reset class specific variables
		self.reset()

		# determine if we are using a data generator or not
		useGen = self.is_data_iter(trainData)

		# if we're using a generator and no steps per epoch raise exception
		if useGen and stepsPerEpoch is None:
			msg = "Using generator without supplying stepsPerEpoch"
			raise Exception(msg)

		# if not then dataset in memory
		elif not useGen:
			# grab the number of samples in training data and calculate stepsPerEpoch
			numSamples = len(trainData[0])
			stepsPerEpoch = int(np.ceil(sampleSize / float(batchSize)))

		# if number of training epoch supplied, compute the training epoch
		if epochs is None:
			epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))

		# compute the number of batch updates that will take place
		numBatchUpdates = epochs * stepsPerEpoch

		# derive the learning rate mulriplier based on the ending learning rate
		# starting learning rate, total number of batch updates
		self.lrMult = (endLR / startLR)**(1.0/numBatchUpdates)

		# create temporary file path to store model weights and save the weights
		self.weightsFile = tempfile.mkstemp()[1]
		self.model.save_weights(self.weightsFile)

		# grab the *original* learning rate, and set the stating learning rate
		origLR = K.get_value(self.model.optimizer.lr)
		K.set_value(self.model.optimizer.lr, startLR)

		# construct a callback that will be called at the end of each batch,
		# so we can increase learning rate as training progresses
		callback = LambdaCallback(on_batch_end=lambda batch, 
			logs:self.on_batch_end(batch, logs))

		# check to see if we are using a data iterator
		if useGen:
			self.model.fit(x=trainData, steps_per_epoch=stepsPerEpoch,
				epochs=epochs, verbose=verbose, callbacks=[callback])

		# otherwise, training data is in memory
		else:
			self.model.fit(
				x=trainData[0], y=trainData[1],
				batch_size=batchSize,
				epochs=epochs,
				callbacks=[callback],
				verbose=verbose)

		# restore the original model weights and learning rate
		self.model.load_weights(self.weightsFile)
		K.set_value(self.model.optimizer.lr, origLR)

	def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
		# grab the learning rate and losses values to plot
		lrs = self.lrs[skipBegin:-skipEnd]
		losses = self.losses[skipBegin:-skipEnd]
		# plot the learning rate vs. loss
		plt.plot(lrs, losses)
		plt.xscale("log")
		plt.xlabel("Learning Rate (Log Scale)")
		plt.ylabel("Loss")
		# if the title is not empty, add it to the plot
		if title != "":
			plt.title(title)




