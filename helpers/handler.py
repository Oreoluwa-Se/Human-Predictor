# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

# function overlap area
# author: Adrian Rosebrock, IOU
# date: 2020-10-04
def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	denom = max(float(boxAArea + boxBArea - interArea), 1e-5)
	iou = interArea / denom 
	# return the intersection over union value
	return iou

# function non maxima suppression
# author: Adrian Rosebrock, Faster nms
# url: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# date: 2020-10-04
def nms(boxes, probs, overlapThresh):
	# if there are no boxes, return empty list
	if len(boxes) == 0:
		return []

	# if bounding boxes are integers convert them to floats
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding box
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of boudning boxes and sort bounding box by probability
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(probs)

	# keep looping while some indexes remain in the indexes list
	while len(idxs) > 0:
		# grab the last index and add to the picked list
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding box
		# and smallest (x, y) for end of bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], x2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the overlap ratio
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from list that have iverlap greater than specified
		idxs = np.delete(idxs, np.concatenate(([last], 
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	return boxes[pick].astype("int")

