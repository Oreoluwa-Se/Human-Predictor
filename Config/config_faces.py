# import the necessary packages
import os

# define path to database
faces_base_path = "Output\\Face_Detector"
face_db = os.path.sep.join([faces_base_path, "total_face.txt"])

# define patht to the output directories [faces/ no face]
positive_Path = os.path.sep.join([faces_base_path, "face"])
negative_Path = os.path.sep.join([faces_base_path, "noface"])

# max proposals when running selective search and inference
max_proposals = 2000
max_proposal_infer = 200

# define nummber of positive and negative images per image
max_pos = 30
max_neg = 10

# input network dims
input_dim = (224, 224)

# path to the output model and label binarizer
model_path = face_db = os.path.sep.join([faces_base_path, "face_detector.h5"])
encod_path = face_db = os.path.sep.join([faces_base_path, "face_encoder.pickle"])

# probability required for positive pred
min_prob = 0.99