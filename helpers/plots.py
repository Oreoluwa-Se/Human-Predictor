# import the necessary pacakages
from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
	def __init__(self, figPath, jsonPath=None, startAt=0):
		# store output path for the figure and path to Json serialized file
		super(TrainingMonitor, self).__init__()
		self.figPath = figPath
		self.jsonPath = jsonPath
		self.startAt = startAt

	# defines what happens at the begining 
	def on_train_begin(self, logs=None):
		# initialize the history dictionary -> contains train_loss, train_accuract,
		# validation_loss, validation_accuracy
		self.H = {}

		# if the Json history path exists, load the training history
		if self.jsonPath is not None:
			if os.path.exists(self.jsonPath):
				self.H = json.loads(open(self.jsonPath).read())

				# check to see if starting epoch was supplied
				if self.startAt > 0:
					# loop over the current entries stored and trim after the
					# start point
					for k in self.H.keys():
						self.H[k] = self.H[k][:self.startAt]


	# defines what happenes at the end of each epoch
	def on_epoch_end(self, epoch, logs={}):
		# loop over all logs and update the loss, accuracy e.t.c
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(float(v))
			self.H[k] = l

		# check to see if the training history should be serialized
		if self.jsonPath is not None:
			f = open(self.jsonPath, "w")
			f.write(json.dumps(self.H))
			f.close()

		# plot at the end of each epoch once we have more than two epchs
		if len(self.H["loss"]) > 1:
			N = np.arange(0, len(self.H["loss"]))
			plt.style.use("ggplot")
			plt.figure()
			plt.plot(N, self.H["loss"], label="train_loss")
			plt.plot(N, self.H["val_loss"], label="val_loss")
			plt.plot(N, self.H["accuracy"], label="train_acc")
			plt.plot(N, self.H["val_accuracy"], label="val_acc")
			plt.title("Training Loss and Accuracy at Epoch {}".format(len(self.H["loss"])))
			plt.xlabel("Epoch#")
			plt.ylabel("Loss/Accuracy")
			plt.legend()

			# save the figurre
			plt.savefig(self.figPath)
			plt.close()

