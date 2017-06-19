import os
import tensorflow as tf 
from PIL import Image
import argparse
import random
import csv 
#import cvutils 
from sklearn.preprocessing import normalize 
from scipy.stats import itemfreq 
from skimage.feature import local_binary_pattern 
import cv2 
import numpy as np


eps=1e-7

# Parameters
path_list = ['../X_data', '../T_data']
lbp_path  = ['../X_data', '../T_data'] # LBP path here
data_list = ["tfrecords/train.tfrecords", "tfrecords/test.tfrecords"]
lbp_list  = ["tfrecords/train_lbp.tfrecords", "tfrecords/test_lbp.tfrecords"]
age = ['child', 'young', 'adult', 'elder']
gender = ['male', 'female']

#
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def data_converter(path, tf_data, args):
	numlist = []
	buff = []
	with tf.python_io.TFRecordWriter(tf_data) as converter:
		for idx_age, itr_age in enumerate(age):
			for idx_gender, itr_gender in enumerate(gender):
				class_label = [idx_age + idx_gender * 4]
				current_path = "{:s}/{:s}/{:s}/".format(path, itr_age, itr_gender)

				n_file = sum(os.path.isfile(os.path.join(current_path, itr_dir)) \
							for itr_dir in os.listdir(current_path))
				numlist.append(n_file)
				if args.verbosity:
					print "{:s}: {:4d} files".format(current_path, n_file)
				for itr_file in os.listdir(current_path):
					print "{:s}: {:4d} files".format(current_path, n_file)

					if itr_file.endswith('.jpg'):
						img_path = current_path + itr_file

						if args.lbp:
							L = LBP(img_path,2,8)
							img =  Image.fromarray(L,mode = 'L')
							#print img
							img_raw = img.tobytes()
							# stream data to the converter
							example = tf.train.Example(features=tf.train.Features(
							feature=
							{ 
								"label"  : _int64_feature(class_label),
								"img_raw": _bytes_feature(img_raw)
							} ))
						
						else:
							img = Image.open(img_path)
							#print img
							img_raw = img.tobytes()
							# stream data to the converter
							example = tf.train.Example(features=tf.train.Features(
							feature=
							{ 
								"label"  : _int64_feature(class_label),
								"img_raw": _bytes_feature(img_raw)
							} ))

						if not args.randomize:
							converter.write(example.SerializeToString())
						else:
							buff.append(example.SerializeToString())
					else:
						continue
		if args.randomize or args.lbp:
			index = random.sample(xrange(len(buff)), len(buff))
			for itr in index:
				converter.write(buff[itr])
	print "{:s}: {:d}".format(path, sum(numlist))


def LBP(train_image,radius, no_points):
	im = cv2.imread(train_image)
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	lbp = local_binary_pattern(im_gray, no_points, radius, method='nri_uniform')
	#(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0,(no_points*(no_points-1)+4)))
	#hist = hist.astype("float")
	#hist /= (hist.sum() + eps)
	#print hist.sum()
	return lbp

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--verbosity", action="store_true",
						help="show info in each directory")
	parser.add_argument("-r", "--randomize", action="store_true",
						help="randomize the order of the data across classes")
	parser.add_argument("-l", "--lbp", action="store_true",
						help="use Local Binary Pattern (LBP) to processing")

	args = parser.parse_args()
	if not os.path.isdir('tfrecords'):
		os.makedirs('tfrecords')

	input_path  = path_list if not args.lbp else lbp_path
	output_name = data_list if not args.lbp else lbp_list
	for idx, itr_path in enumerate(input_path):
		data_converter(itr_path, output_name[idx], args)
