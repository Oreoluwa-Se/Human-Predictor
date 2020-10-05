# import the necessary packages
from os import path
import numpy as np


# define location for dlib weight
d_weight = "D:\\Artificial_Intelli\\Courses\\PyimageSearch\\Deep_Learning\\Dataset\\Weights\\mmod_human_face_detector.dat"

# base path
base_Path = "D:\\Artificial_Intelli\\Courses\\PyimageSearch\\Deep_Learning\\Projects\\FD-AGR"

# face detection scheme
dataset_Path = "D:\\Artificial_Intelli\\Courses\\PyimageSearch\\Deep_Learning\\Dataset\\Faces"

# image and annotation path
image_Path = path.sep.join([dataset_Path, "Images"])
annotations_Path = path.sep.join([dataset_Path, "Annotations"])

# dataset names
dataset_Names = ["Aligned", "Cropped", "ISAFE"]


# define the output path
output_Path = path.sep.join([base_Path, "Output"])

# database storage
total_emotion = path.sep.join([output_Path, "emotion_Detector", "total_emotion.txt"])
total_face = path.sep.join([output_Path, "Face_Detector", "total_face.txt"])
total_race = path.sep.join([output_Path, "Race_Detector", "total_race.txt"])
total_age = path.sep.join([output_Path, "Age_Detector", "total_age.txt"])
total_sex = path.sep.join([output_Path, "Sex_Detector", "total_sex.txt"])


# age bins
bin_age = ["(0, 2)", "(4, 6)", "(8, 12)", "(15, 20)", "(21, 24)", "(25, 32)",
	"(33, 37)", "(38, 43)", "(44, 47)", "(48, 53)", "(54, 59)", "(60, 100)", "100+"]

# ISAFE Manipulation emotion swap
emotion = [3,4,5,1,2,0,8,7]

# sex classification
male=[1, 2, 4, 5, 7, 8, 10, 15, 17, 18, 19, 22, 27, 30, 32, 33, 34, 35, 37, 39, 41]
fem = [3, 6, 9, 23, 24, 25, 26, 28, 29, 31, 36, 38, 40, 42, 43, 44]