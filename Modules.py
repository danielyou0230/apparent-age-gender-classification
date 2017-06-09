import cv2
import numpy as np
import os
import pandas
import dlib
import imutils
import time
from scipy.ndimage.filters import gaussian_filter
import random
from scipy import ndimage
import argparse

# To-be-done
# Make loose detection on classifier
# Double check the faces acquire by classifier
# Make Face bigger for landmark

mainPath = '../Dataset'
prepPath = '../Preprocessed'
xPath = '../X_data'
tPath = '../T_data'
age = ['child', 'young', 'adult', 'elder']
gender = ['male', 'female']
xml_face_classifier = '../haarcascades/haarcascade_frontalface_default.xml'
dat_face_landmark   = '../shape_predictor_68_face_landmarks.dat'

# Modules below
def cleanup_workspace(path2remove, ext='.jpg'):
	file_list = os.listdir(path2remove)
	for item in file_list:
		if item.endswith(ext):
			os.remove(os.path.join(path2remove, item))

def load_data(file, ravel=False):
	data = pandas.read_csv(file, header=None)
	data = data.as_matrix()
	return data if not ravel else data.ravel()

def get_augment_list(filepath):
	file_list = list()
	for itr in os.listdir(filepath):
		if itr.startswith('training') and itr.endswith('.csv') and \
		   ('aug' in itr) and (not 'target' in itr):
			file_list.append(filepath + itr)
	#print file_list
	return file_list

def get_dataInfo(filepath):
	#print [itr_dir for itr_dir in os.listdir('Dataset/adult')]
	return sum(os.path.isfile(os.path.join(filepath, itr_dir)) \
							for itr_dir in os.listdir(filepath))

def rect_to_bb(rect, max_size, file):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	# adjustment: make the capture image bigger or smaller
	adjustment = 0.00

	x = rect.left() 
	x_expansion = x * (1 - adjustment)
	x = x_expansion if x_expansion > 0 and x_expansion < max_size[0] else \
		max_size[0] if x_expansion > max_size[0] else \
		0
	
	y = rect.top() 
	y_expansion = y * (1 - adjustment)
	y = y_expansion if y_expansion > 0 and y_expansion < max_size[1] else \
		max_size[1] if y_expansion > max_size[1] else \
		0
	
	w = rect.right() - x
	w_expansion = w * (1 + adjustment) + x
	# coordinate after expansion: w * (1 + adjustment) + x 
	w = (w_expansion - x) if w_expansion > 0 and w_expansion < max_size[0] else \
		(max_size[0] - x) if w_expansion > max_size[0] else \
		w

	h = rect.bottom() - y
	h_expansion = h * (1 + adjustment) + y
	h = (h_expansion - y) if h_expansion > 0 and h_expansion < max_size[1] else \
		(max_size[1] - y) if h_expansion > max_size[1] else \
		h

	if h == 0 or w == 0:
		print file
		return (int(x), int(y), int(w), int(h))

	ratio = 1.0 * w / h
	if ratio < 0.9:
		w = h
	elif ratio > pow(0.9, -1):
		h = w

	return (int(x), int(y), int(w), int(h))

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

def face_detect_classifier(image, face_cascade, file):
	# Load image as greyscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Use OpenCV Classifier the find faces
	cascade_faces = face_cascade.detectMultiScale(image, 
												  scaleFactor=1.10,
												  minNeighbors=5)
	for (x, y, w, h) in cascade_faces:
		cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)

	bFace = True if len(cascade_faces) != 0 else False

	# Check Face availability
	if bFace:
		# Return face coordinates
		# Multi-Face: Choose the Largest Area
		if len(cascade_faces) > 1:
			area = 0
			face_idx = 0
			# Pick the largest area
			for (idx, (x, y, w, h)) in enumerate(cascade_faces):
				if (w * h) > area:
					area = w * h
					face_idx = idx
			
			faces = cascade_faces[face_idx]
		
		# Single-Face: just return
		else:
			faces = cascade_faces[0]
		
		# Square out the face
		x, y, w, h = faces
		cv2.rectangle(image, (x,y), (x+w,y+h), (0, 0, 255), 3)

	# Return empty coordinates if no faces detected
	else:
		print "Classifier Failed: {:s}".format(file)
		faces = list()
	return bFace, faces

def facial_landmark_detection(image, detector, predictor, file):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_size = gray.shape
	landmark_faces = detector(gray, 1)

	faces = list()
	area = 0
	face_idx = 0
	bItr = False
	for (idx, landmark_faces) in enumerate(landmark_faces):
		shape = predictor(gray, landmark_faces)
		shape = shape_to_np(shape)
		(x, y, w, h) = rect_to_bb(landmark_faces, img_size, file)
		
		if (w * h) > area:
			area = w * h
			faces = [x, y, w, h]
			bItr = True
		#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
		#cv2.putText(image, "Face #{}".format(idx + 1), (x - 10, y - 10), \
		#           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		#for (x, y) in shape:
		#   cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	return bItr, faces

def face_landmark_Preliminary():
	# Preliminary process (To be improved)
	# Prepare for dlib facial landmark
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(dat_face_landmark)
	#face_cascade = cv2.CascadeClassifier(xml_face_classifier)

	undetectLst = list()
	#false_Lst = list()
	max_size = [0, 0, 'null']
	min_size = [500, 500, 'null']
	tStart = time.time()

	for itr_age in age:
		for itr_gender in gender:
			curr_dir = "{:s}/{:s}/{:s}/".format(mainPath, itr_age, itr_gender)
			numfile = get_dataInfo(curr_dir)
			not_detected = 0
			#false_detect = 0
			
			#for itr_file in os.listdir(curr_dir):
			for itr in range(numfile / 2):
				#if itr_file.endswith('.jpg'):
				itr_file = "{:d}.jpg".format(itr)
				file = curr_dir + itr_file
				image = cv2.imread(curr_dir + itr_file)
				image = imutils.resize(image, width=500)
				bFace, faces = facial_landmark_detection(image, detector, predictor, file)
				
				if not bFace:
					#print curr_dir + itr_file
					undetectLst.append(curr_dir + itr_file)
					not_detected += 1
					continue
				x, y, w, h = faces
				
				# Recording the max, min size of the crop images for rescaling
				if (max_size[0] * max_size[1]) < (w * h):
					max_size = [w, h, curr_dir + itr_file]
				
				if (min_size[0] * min_size[1]) > (w * h):
					min_size = [w, h, curr_dir + itr_file]
				crop_img = image[y:y + h, x:x + w]

				cv2.imwrite("{:s}/{:s}/{:s}/face_{:s}" \
							.format(prepPath, itr_age, itr_gender, itr_file), \
							crop_img)
				#crop_img = image[y:y + h, x:x + w]
				#cv2.imshow(file, crop_img)
				#cv2.imshow(file, image)
				#cv2.waitKey(0)
				#cv2.destroyAllWindows() 
				#elif bDupFace:
				#   false_Lst.append(curr_dir + itr_file)
				#   false_detect += 1

				itr += 1 

			print "{:s}: {:4d}/{:4d}".format(curr_dir, not_detected, numfile / 2)
			#print " - false detected: {:4d}/{:4d}".format(false_detect, numfile / 2)
	
	tEnd = time.time()
	print "Time Elapse: {:.2f} sec".format(tEnd - tStart)

	print "max size: {:s}".format(max_size)
	print "min size: {:s}".format(min_size)

	df = pandas.DataFrame(undetectLst) 
	df.to_csv('../dlib_undetected.csv', header=False, index=False)
	#df = pandas.DataFrame(false_Lst)
	#df.to_csv('../dlib_falsedetected.csv', header=False, index=False)

def generate_mode(amount, sample_size, mode):
	# Generate the index of the image and the corresponding augmentation mode
	sample_list = random.sample(xrange(amount), sample_size)
	sample_type = [np.random.choice(mode, 1) for itr in range(sample_size)]
	#for itr in mode:
	#	print "{:1d}: {:3d}".format(itr, sample_type.count(itr))
	sample = np.hstack([np.vstack(sample_list), np.vstack(sample_type)])
	sample = sample[sample[:, 0].argsort()]
	return sample

def export2csv(blur=True, sigma=2.0, hflip=True, rotate=False, 
			   img_size=100, sample_size=100):
	# Export and resize the image to cvs files with data augmentation included
	data = list()
	aug_br = list()
	aug_hf = list()
	target = list()
	amount = list()
	testing_data = list()
	testing_targ = list()

	mode = list()
	# The criterion of different data augmentations 
	mode_info  = [True, blur, hflip, rotate]
	for index, itr in enumerate(mode_info):
		if itr:
			mode.append(index)
		else:
			continue

	# Load and distribute the data to the corresponding group
	for (age_index, itr_age) in enumerate(age):
		for (gen_index, itr_gender) in enumerate(gender):
			curr_dir = "{:s}/{:s}/{:s}/".format(prepPath, itr_age, itr_gender)
			numfile = get_dataInfo(curr_dir)

			# Generate the index and the mode of the image to be testing data
			sample = generate_mode(numfile, sample_size, mode)
			# Keep track of the amount of testing data for each data augmentation
			amount_list = [0] * 4
			normal_list = [0] * 4
			for index, itr_file in enumerate(os.listdir(curr_dir)):
				is_testing = False
				if itr_file.endswith('.jpg'):
					image = cv2.imread(curr_dir + itr_file, 0)
					image = cv2.resize(image, (img_size, img_size))
					# Original image
					if index in sample[:, 0]:
						sample_index = list(sample[:, 0]).index(index)
						if sample[sample_index, 1] == 0:
							testing_data.append(image.reshape(1, image.size))
							is_testing = True
							amount_list[0] += 1
						else:
							data.append(image.reshape(1, image.size))
							normal_list[0] += 1
							target.append([age_index + gen_index * 4])
					else:
						data.append(image.reshape(1, image.size))
						normal_list[0] += 1
						target.append([age_index + gen_index * 4])
					# Data Augmentation:
					## Gaussian Blurred 
					if blur and (index in sample[:, 0]):
						image_blur = gaussian_filter(input=image, sigma=sigma)
						sample_index = list(sample[:, 0]).index(index)
						if sample[sample_index, 1] == 1:
							testing_data.append(image_blur.reshape(1, image_blur.size))
							is_testing = True
							amount_list[1] += 1
						else:
							aug_br.append(image_blur.reshape(1, image_blur.size))
							normal_list[1] += 1
							target.append([age_index + gen_index * 4])
					elif blur:
						image_blur = gaussian_filter(input=image, sigma=sigma)
						aug_br.append(image_blur.reshape(1, image_blur.size))
						normal_list[1] += 1
						target.append([age_index + gen_index * 4])
					## Flip and Rotate 
					### hflip and not vflip: output horizontal flipped image
					#criterion = (hflip and not vflip) or (hflip and hvsplit)
					if hflip and (index in sample[:, 0]):
						image_flip = np.fliplr(image)
						sample_index = list(sample[:, 0]).index(index)
						if sample[sample_index, 1] == 2:
							testing_data.append(image_flip.reshape(1, image_flip.size))
							is_testing = True
							amount_list[2] += 1
						else:
							aug_hf.append(image_flip.reshape(1, image_flip.size))
							normal_list[2] += 1
							target.append([age_index + gen_index * 4])
					elif hflip:
						image_flip = np.fliplr(image)
						aug_hf.append(image_flip.reshape(1, image_flip.size))
						normal_list[2] += 1
						target.append([age_index + gen_index * 4])
					#criterion = hflip and vflip and not hvsplit
					if rotate and (index in sample[:, 0]):
						image_flip = np.rot90(image, k=2)
						sample_index = list(sample[:, 0]).index(index)
						if sample[sample_index, 1] == 3:
							testing_data.append(image_flip.reshape(1, image_flip.size))
							is_testing = True
							amount_list[3] += 1
						else:
							aug_rt.append(image_flip.reshape(1, image_flip.size))
							normal_list[3] += 1
							target.append([age_index + gen_index * 4])
					elif rotate:
						image_flip = np.rot90(image, k=2)
						aug_rt.append(image_flip.reshape(1, image_flip.size))
						normal_list[3] += 1
						target.append([age_index + gen_index * 4])

					# Append the target to the list
					if is_testing:
						# Class token
						testing_targ.append([age_index + gen_index * 4])

				else: 
					continue
			# Append information for each class to the list
			amount.append([itr_gender, itr_age, age_index + gen_index * 4, numfile, 
						  sum(normal_list),
						  normal_list[0], normal_list[1], 
						  normal_list[2], normal_list[3], 
						  sum(amount_list), 
						  amount_list[0], amount_list[1], 
						  amount_list[2], amount_list[3], 
						  sigma, "{:d}x{:d}".format(img_size, img_size)])
	df.to_csv('../CSV_Data/processed_amount.csv', index=False)

def clear_cache():
	for (age_index, itr_age) in enumerate(age):
		for (gen_index, itr_gender) in enumerate(gender):
			curr_dir = "{:s}/{:s}/{:s}/".format(prepPath, itr_age, itr_gender)
			x_path = "{:s}/{:s}/{:s}".format(xPath, itr_age, itr_gender)
			t_path = "{:s}/{:s}/{:s}".format(tPath, itr_age, itr_gender)
			cleanup_workspace(x_path) if os.path.isdir(x_path) else os.makedirs(x_path)
			cleanup_workspace(t_path) if os.path.isdir(t_path) else os.makedirs(t_path)

def data_augment(blur=True, sigma=[2.0], hflip=True, rotate=False, 
				 img_size=100, sample_size=100):
	# Export and resize the image to cvs files with data augmentation included
	data = list()
	target = list()
	amount = list()
	testing_targ = list()
	mode = list()
	# The criterion of different data augmentations 
	mode_info  = [True, blur, hflip, rotate]
	for index, itr in enumerate(mode_info):
		if itr:
			mode.append(index)
		else:
			continue
	
	clear_cache()
	# Load and distribute the data to the corresponding group
	for (age_index, itr_age) in enumerate(age):
		for (gen_index, itr_gender) in enumerate(gender):
			curr_dir = "{:s}/{:s}/{:s}/".format(prepPath, itr_age, itr_gender)
			x_path = "{:s}/{:s}/{:s}".format(xPath, itr_age, itr_gender)
			t_path = "{:s}/{:s}/{:s}".format(tPath, itr_age, itr_gender)
			
			# If the path exist, clear previous work, otherwise create the path
			cleanup_workspace(x_path) if os.path.isdir(x_path) else os.makedirs(x_path)
			cleanup_workspace(t_path) if os.path.isdir(t_path) else os.makedirs(t_path)

			numfile = get_dataInfo(curr_dir)
			#print "{:s}: {:4d}, {:4d}".format(curr_dir, numfile, 
			#								  numfile * sum(mode_info) - sample_size)
			# Generate the index and the mode of the image to be testing data
			sample = generate_mode(numfile, sample_size, mode)
			# Keep track of the amount of testing data for each data augmentation
			amount_list = [0] * sum(mode_info)
			normal_list = [0] * sum(mode_info)
			for index, itr_file in enumerate(os.listdir(curr_dir)):
				is_testing = False
				if itr_file.endswith('.jpg'):
					token = [age_index + gen_index * 4]
					image = cv2.imread(curr_dir + itr_file, 0)
					image = cv2.resize(image, (img_size, img_size))
					# Original image
					if index in sample[:, 0]:
						sample_index = list(sample[:, 0]).index(index)
						if sample[sample_index, 1] == 0:
							cv2.imwrite("{:s}/{:s}".format(t_path, itr_file), image)
							is_testing = True
							amount_list[0] += 1
						else:
							cv2.imwrite("{:s}/{:s}".format(x_path, itr_file), image)
							normal_list[0] += 1
							target.append(token)
					else:
						cv2.imwrite("{:s}/{:s}".format(x_path, itr_file), image)
						normal_list[0] += 1
						target.append(token)
					# Data Augmentation:
					## Gaussian Blurred 
					if blur and (index in sample[:, 0]):
						for itr_sigma in sigma:
							testfile  = "{:s}/b_{:1.1f}_{:s}".format(t_path, itr_sigma, itr_file)
							trainfile = "{:s}/b_{:1.1f}_{:s}".format(x_path, itr_sigma, itr_file)
							image_blur = gaussian_filter(input=image, sigma=itr_sigma)
							sample_index = list(sample[:, 0]).index(index)
							if sample[sample_index, 1] == 1 and not is_testing:
								cv2.imwrite(testfile, image_blur)
								is_testing = True
								amount_list[1] += 1
							else:
								cv2.imwrite(trainfile, image_blur)
								normal_list[1] += 1
								target.append(token)
					elif blur:
						for itr_sigma in sigma:
							image_blur = gaussian_filter(input=image, sigma=itr_sigma)
							cv2.imwrite("{:s}/b_{:1.1f}_{:s}" \
										.format(x_path, itr_sigma, itr_file),  \
										image_blur)
							normal_list[1] += 1
							target.append(token)
					## Flip and Random Rotate 
					if hflip and (index in sample[:, 0]):
						image_flip = np.fliplr(image)
						sample_index = list(sample[:, 0]).index(index)
						if sample[sample_index, 1] == 2:
							cv2.imwrite("{:s}/h_{:s}".format(t_path, itr_file), image_flip)
							is_testing = True
							amount_list[2] += 1
						else:
							cv2.imwrite("{:s}/h_{:s}".format(x_path, itr_file), image_flip)
							normal_list[2] += 1
							target.append(token)
					elif hflip:
						image_flip = np.fliplr(image)
						cv2.imwrite("{:s}/h_{:s}".format(x_path, itr_file), image_flip)
						normal_list[2] += 1
						target.append(token)
					#
					##if rotate and (index in sample[:, 0]):
					##  image_flip = np.rot90(image, k=2)
					##  sample_index = list(sample[:, 0]).index(index)
					##  if sample[sample_index, 1] == 3:
					##      cv2.imwrite("{:s}/{:s}".format(t_path, itr_file), image_flip)
					##      is_testing = True
					##      amount_list[3] += 1
					##  else:
					##      cv2.imwrite("{:s}/{:s}".format(x_path, itr_file), image_flip)
					##      normal_list[3] += 1
					##      target.append([age_index + gen_index * 4])
					##elif rotate:
					##  image_flip = np.rot90(image, k=2)
					##  cv2.imwrite("{:s}/{:s}".format(x_path, itr_file), image_flip)
					##  normal_list[3] += 1
					##  target.append([age_index + gen_index * 4])

					# Append the target to the list
					if is_testing:
						testing_targ.append(token)
						#for itr in range(sum(mode_info) - 1):
						#	target.append(token)
					#else:
					#	for itr in range(sum(mode_info)):
					#		target.append(token)
				else: 
					continue
			# Append information for each class to the list
			#print normal_list
			amount.append([itr_gender, itr_age, age_index + gen_index * 4, numfile, 
					  sum(normal_list), normal_list[0], normal_list[1], normal_list[2], 
					  sum(amount_list), amount_list[0], amount_list[1], amount_list[2], 
					  "{:d}x{:d}".format(img_size, img_size)])

	df = pandas.DataFrame(np.reshape(np.ravel(target), (len(target), 1)))
	print np.ravel(target).shape
	df.to_csv("{:s}/X_target.csv".format(xPath), header=False, index=False)
	df = pandas.DataFrame(np.vstack(testing_targ))
	df.to_csv("{:s}/T_target.csv".format(tPath), header=False, index=False)
	df = pandas.DataFrame(np.vstack(amount), 
						  columns=['Gender', 'Age', 'Class', 'Org_Amt',
								   'Training', 'Tr_Normal', 'Tr_Blur', 'Tr_hflip', 
								   'Testing' , 'Te_Normal', 'Te_Blur', 'Te_hflip', 
								   'img_size'])
	df.to_csv('../processed_amount.csv', index=False)

def evaluate_result(prediction, target, numClass):
	numData = len(prediction)
	err = [0] * numClass
	result = [0] * numData
	for itr in range(numData):
		if prediction[itr] != target[itr]:
			err[target[itr] - 1] += 1
			result[itr] = 1
	
	print "Error rate: {:.2f}% ({:4d}/{:4d})" \
		  .format(100.0 * sum(err) / numData, \
				  sum(err), numData)
	
	info_str = " - "
	for itr in range(numClass):
		if itr != numClass - 1:
			info_str += "Class{:2d}: {:3d}, ".format(itr + 1, err[itr])
		else:
			info_str += "Class{:2d}: {:3d}".format(itr + 1, err[itr])
	print info_str

# Debugging code
def debug_face_classifier(file):
	face_cascade = cv2.CascadeClassifier(xml_face_classifier)
	image = cv2.imread(file)
	
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(image, 1.20, 5, minSize = (50, 50))
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
		#roi_gray = gray[y:y+h, x:x+w]
		#roi_color = image[y:y+h, x:x+w]

	cv2.imshow('Image', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def debug_face_landmark(file, output=False, output_name='output'):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(dat_face_landmark)

	image = cv2.imread(file)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	img_size = gray.shape

	faces = detector(gray, 1)
	for (i, itr_face) in enumerate(faces):
		shape = predictor(gray, itr_face)
		shape = shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = rect_to_bb(itr_face, img_size, file)
		#print "landmark: ({:d}, {:d}) ({:d}, {:d})".format(x, y, w, h)
		
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	 
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	# show the output image with the face detections + facial landmarks
	cv2.imshow(file, image)
	cv2.waitKey(0)
	if output:
		cv2.imwrite("../" + str(output_name + 1) + '.jpg', image)
	cv2.destroyAllWindows()

def debug_Data_Augmentation(blur=False, sigma=1.0, hflip=False, vflip=False, hvsplit=False, randbright=False):
	image = cv2.imread('Dataset/young/female/180.jpg', 0)
	#image = cv2.imread('Dataset/young/female/285.jpg', 0)
	#image = cv2.resize(image, (100, 100))
	cv2.imshow('Image', image)

	# Data Augmentation:
	# Gaussian Blurred 
	if blur:
		cv2.imshow('Blur', gaussian_filter(input=image, sigma=sigma))
		#cv2.imwrite("Blur_{:1.1f}.jpg".format(sigma), 
		#			gaussian_filter(input=image, sigma=sigma))
		cv2.imwrite("../xBlur_{:1.1f}.jpg".format(sigma), 
					gaussian_filter(input=image, sigma=sigma))
	# Flip and Rotate
	if (hflip and not vflip) or (hflip and hvsplit):
		cv2.imshow('hflip', np.fliplr(image))
		cv2.imwrite("../hflip.jpg", np.fliplr(image))
	if (vflip and not hflip) or (vflip and hvsplit):
		cv2.imshow('vflip', np.flipud(image))
		cv2.imwrite("../vflip.jpg", np.flipud(image))
	if hflip and vflip and not hvsplit:
		cv2.imshow('rot 180', np.rot90(image, k=2))
		cv2.imwrite("../rot2k.jpg", np.rot90(image, k=2))
	
	#if randbright:
	#   to_brightest = 255 - image.max()
	#   to_darkest   =  -1 * image.min()
	#   print "Brightest: {:d}, To-the-Roof : {:d}".format(image.max(), to_brightest)
	#   print "Darkest  : {:d}, To-the-Floor: {:d}".format(image.min(), to_darkest)
	#   disturbance = random.randint(to_darkest, to_brightest)
	#   print "Disturbance: {:d}".format(disturbance)
	#   image = (image  -18).astype(np.uint8)
	#   #print "Random Factor: {:1.3f}".format(disturb_factor)
	#   #image = (image * disturb_factor).astype(np.uint8)
	#   cv2.imshow('brighter', image)

	cv2.waitKey(0)
	cv2.destroyAllWindows() 

def debug_analyse_image_texture(file, sigma=1.0):
	image = cv2.imread(file, 0)
	blur  = gaussian_filter(input=image, sigma=sigma)
	cv2.imshow('Image', image - blur)
	#analysis = ndimage.gaussian_gradient_magnitude(image, sigma=sigma)
	#cv2.imshow('Analysis', analysis * 10)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
##########################################
#debug_face_classifier('Dataset/young/male/178.jpg')
#file_list = ['Dataset/adult/female/79.jpg', 
#            'Dataset/elder/male/26.jpg', 
#            'Dataset/adult/male/153.jpg', 
#            'Dataset/child/female/69.jpg', 
#            'Dataset/child/male/66.jpg', 
#            'Dataset/young/female/98.jpg', 
#            'Dataset/elder/female/45.jpg', 
#            'Dataset/young/male/134.jpg', 
#            'Dataset/adult/male/47.jpg', 
#            'Dataset/elder/female/5.jpg']
#for index, itr in enumerate(file_list):
#   debug_face_landmark(itr, output=True, output_name=index)
#debug_face_landmark('Dataset/child/female/69.jpg', output=True, output_name=4)

#debug_Data_Augmentation(blur=False, sigma=0.7, hflip=True, vflip=True, hvsplit=True,
#                       randbright=True)
#debug_analyse_image_texture('150.jpg', sigma=7)

if __name__ == "__main__":
	#parser = argparse.ArgumentParser()
	#parser.add_argument("-v", "--verbosity", action="count",
	#					help="show info in each directory")
	#face_landmark_Preliminary()
	#export2csv(blur=True, sigma=2.0, hflip=True, vflip=False, \
	#          hvsplit=True, img_size=100, sample_size=100)
	data_augment(blur=True, sigma=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], \
				 hflip=True, rotate=False, img_size=128, sample_size=100)
	#debug_analyse_image_texture(file='Dataset/adult/female/79.jpg', sigma=1.0)
