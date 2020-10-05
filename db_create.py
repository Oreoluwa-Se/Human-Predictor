# import the necessary packages
from Config import config_partition as config
from datetime import datetime
from imutils import paths
import argparse
import dlib
import cv2
import os

# function for age def
def age_check(config, age):
	# check to see if age is already categorized
	if age[-1] == ")":
		# check outliers
		if age == "(27, 32)":
			return "(25, 32)"

		elif age == "(8, 23)":
			return "(8, 12)"

		elif age == "(38, 42)":
			return "(38, 43)"

		else:
			return age

	# age not categorized yet
	else:
		# convert to integer
		age = int(age)

		# loop through the age bins
		for binx in range(0, len(config.bin_age) -1):
			# extract bin
			bins = config.bin_age[binx]
			# extract the upper range of the bin bracket
			bin_s = bins.strip().split(",")
			bin_upper =  int(bin_s[1][1:-1])

			# if less than the upper we have found the category
			if age <= bin_upper:
				return bins

		# else return the final bin
		return config.bin_age[-1]


# initialize the dlib cnn with weights
dlib_cnn = dlib.cnn_face_detection_model_v1(config.d_weight)

print("[INFO] Opening dataset...")
# open all dataset to be written into
output_face = open(config.total_face, "w")
output_race = open(config.total_race, "w")
output_age = open(config.total_age, "w")
output_sex = open(config.total_sex, "w")
output_emo = open(config.total_emotion, "w")

# start time
start = datetime.now()

# for all the images we have in out dataset
for dataset in config.dataset_Names:
	# annotation path location - all except cropped
	if dataset == "Aligned":
		print("[INFO] Manipulating the Aligned dataset...")
		# loop through all the available folds
		for i in range(1, 5):
			# name of annotation
			fold_name = "fold_" + str(i) + "_data.txt"
			# path to annotation
			fold_path = os.path.sep.join([config.annotations_Path, dataset, fold_name])
			# open the annotation
			rows = open(fold_path).read()
			# split everything except the first line
			rows = rows.strip().split("\n")[1:]

			# loop through every row
			for row in rows:
				# split the rows
				row = row.split("\t")
				# obtain necessary data
				(userID, imagetag, face_id, age, sex, x, y, dx, dy) = row[:9]
				# image name
				imageName="landmark_aligned_face.{}.{}".format(face_id, imagetag)
				# build the image path
				imagePath = os.path.sep.join([config.image_Path, dataset, userID, 
					imageName])

				#bounding box x
				endx = str(int(x) + int(dx))
				endy = str(int(y) + int(dy))
				bbox = x + "\t" + y + "\t" + endx + "\t" + endy

				# copy into files
				face_write = imagePath +"\t"+ bbox +"\n"
				output_face.write(face_write)

				# check if there is a valid sex category
				if (sex == "m") or (sex == "f"):
					sex_write = imagePath +"\t"+ bbox +"\t" + sex + "\n"
					output_sex.write(sex_write)

				# check if the age is not none
				if age != "None":
					# see if to categorize or not
					age_write = imagePath +"\t"+ bbox +"\t" + age_check(config, age) + "\n"
					output_age.write(age_write)

	elif dataset == "Cropped":
		print("[INFO] Manipulating the Cropped dataset...")
		# obtain all image paths
		data_loc = os.path.sep.join([config.image_Path, dataset])
		rows = list(paths.list_images(data_loc))

		# loop through every row
		for row in rows:
			# read image
			image_ = cv2.imread(row)

			# obtain bounding box from image
			f_face = dlib_cnn(image_, 1)

			for face in f_face:
				# bounding box
				x = str(face.rect.left())
				y = str(face.rect.top())
				endx = str(face.rect.right())
				endy = str(face.rect.bottom())

				# collated bounding box
				bbox = x + "\t" + y + "\t" + endx + "\t" + endy

				# split the row
				name = row.split("\\")[-1]

				# obtain the age, sex, and race
				(age, sex, race) = name.split("_")[:3]

				# copy into files
				face_write = row +"\t"+ bbox +"\n"
				output_face.write(face_write)

				# check if the age is not none
				if age != "None":
					# see if to categorize or not
					age_write = row +"\t"+ bbox +"\t" + age_check(config, age) + "\n"
					output_age.write(age_write)

				# check if there is a valid sex category
				if (int(sex) == 1):
					sex_write = imagePath +"\t"+ bbox +"\t" + "f" + "\n"
					output_sex.write(sex_write)
				else:
					sex_write = row +"\t"+ bbox +"\t" + "m" + "\n"
					output_sex.write(sex_write)

				# race section
				if len(race) < 2:
					race_ = "N/A"
					if int(race) == 0:
						race_ = "White"
					elif int(race) == 1:
						race_ = "Black"
					elif int(race) == 2:
						race_ = "Asian"
					elif int(race) == 3:
						race_ = "Indian"

					# write to file
					race_write = row +"\t"+ bbox +"\t" + race_ + "\n"
					output_race.write(race_write)

	elif dataset == "ISAFE":
		print("[INFO] Manipulating the ISAFE dataset...")
		# obtain all annotations
		data_loc = os.path.sep.join([config.annotations_Path, dataset, 
			"Annotations", "psy-annotation.csv"])
		# open the annotation
		annot = open(data_loc).read()
		# split everything except the first line
		annot = annot.strip().split("\n")
		
		# obtain all image paths
		data_loc = os.path.sep.join([config.image_Path, dataset])
		rows = list(paths.list_images(data_loc))

		# for each row
		for row in rows:
			print(row)
			# read image
			image_ = cv2.imread(row)
			# obtain bounding box
			f_face = dlib_cnn(image_, 1)
			for face in f_face:
				# bounding box
				x = str(face.rect.left())
				y = str(face.rect.top())
				endx = str(face.rect.right())
				endy = str(face.rect.bottom())

				# collated bounding box
				bbox = x + "\t" + y + "\t" + endx + "\t" + endy
				# copy into files
				face_write = row +"\t"+ bbox +"\n"
				output_face.write(face_write)

			# obtain the extract emotion and sex
			row_tag = row.split("\\")[-3:]
			# emotion tag
			emo_tag = row_tag[0] + "/" + row_tag[1]

			# loop through the options
			for ind in range(0, len(annot)):
				# if the tag is found
				if emo_tag in annot[ind]:
					var = annot[ind].split(",")
					# emotion indicator
					var = int(var[1])
					emo = config.emotion[var - 1]
					# write to file
					emo_write = row +"\t"+ bbox +"\t" + str(emo) + "\n"
					output_emo.write(emo_write)

			# obtain sex
			sex = "f"
			tag = row_tag[0]
			if int(tag[-1]) in config.male:
				sex = "m"

			sex_write = row +"\t"+ bbox +"\t" + sex + "\n"
			output_sex.write(sex_write)

print("[INFO] Closing database...")
# close all database
output_face.close()
output_race.close()
output_age.close()
output_sex.close()
output_emo.close()

# start time
elapsedTime = (datetime.now() - startTime).seconds
print("Total run time {} seconds...".format(elapsedTime))