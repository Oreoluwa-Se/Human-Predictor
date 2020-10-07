# Use dlib to find bounding box of images that aren't currently 
# import the necessary packages
from Config import config_faces as config
from helpers.handler import iou, nms
from datetime import datetime
from imutils import paths
import dlib
import numpy as np
import cv2
import os

# create the positive and negative image path 
for dirPath in (config.positive_Path, config.negative_Path):
	# if directory doesn't exist create it
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)

# grab database of processed images
print("[INFO] Loading the face annotations...")
# open the annotation
rows = open(config.face_db).read()
# split in new line | rows -> [image location, startx, starty, endx, endy]
rows = rows.strip().split("\n")

# initialize the dlib cnn with weights
dlib_cnn = dlib.cnn_face_detection_model_v1(config.d_weight)

# loop through each image -> 
for (count, row) in enumerate(rows):
	print("[INFO] Processing image {} of {}...".format(count + 1, len(rows)))
	
	# extract contents
	(imagePath, startx, starty, endx, endy) = row.split("\t")
	
	# load image
	image = cv2.imread(imagePath)

	# extract the file name
	file_ = imagePath.split("\\")[-1]
	spot_find = file_.rfind(".")
	file_=file_[:spot_find]

	# image dimensions
	(h, w) = image.shape[:2]
	
	if "landmark_aligned_face" in file_:
		# obtain bounding box from image
		f_face = dlib_cnn(image, 1)

		if len(f_face) > 0:
			for face in f_face:
				# bounding box
				startx = int(max(0, 0.8*face.rect.left()))
				starty = int(max(0, 0.8*face.rect.top()))
				endx = int(min(w, 1.25*face.rect.right()))
				endy = int(min(h, 1.25*face.rect.bottom()))
		else:
			startx = 0
			starty = 0
			endx = w
			endy = h

	else:
		# remove boxes outside boundary
		startx = int(max(0, 0.8*int(startx)))
		starty = int(max(0, 0.8*int(starty)))
		endx = int(min(w, 1.25*int(endx)))
		endy = int(min(h, 1.25*int(endy)))

	# extract the region and derive output path
	roi = image[starty:endy, startx:endx]
	file_name = "{}.png".format(file_)
	outputPath = os.path.sep.join([config.positive_Path, file_name])

	# if roi and output path exist. resize the image to given input dimensions
	if roi is not None and outputPath is not None:
		roi = cv2.resize(roi, config.input_dim, interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(outputPath, roi)
