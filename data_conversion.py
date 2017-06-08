import os
import tensorflow as tf 
from PIL import Image
import argparse

# Parameters
path_list = ['../X_data', '../T_data']
data_list = ["train.tfrecords", "test.tfrecords"]
age = ['child', 'young', 'adult', 'elder']
gender = ['male', 'female']

#
def data_converter(path, tf_data, verbose):
	with tf.python_io.TFRecordWriter(tf_data) as converter:
		for idx_age, itr_age in enumerate(age):
			for idx_gender, itr_gender in enumerate(gender):
				class_label = [idx_age + idx_gender * 4]
				current_path = "{:s}/{:s}/{:s}/".format(path, itr_age, itr_gender)
				
				n_file = sum(os.path.isfile(os.path.join(current_path, itr_dir)) \
							for itr_dir in os.listdir(current_path))
				if verbose:
					print "{:s}: {:4d} files".format(current_path, n_file)
				for itr_file in os.listdir(current_path):
					if itr_file.endswith('.jpg'):
						img_path = current_path + itr_file
						img = Image.open(img_path)
						# convert to tf format
						#dic_label = tf.train.Int64List(value=class_label)
						#dic_data  = tf.train.BytesList(value=img.tobytes())
						# stream data to the converter
						example = tf.train.Example(features=tf.train.Features(
						feature=
						{ 
							"label"  : tf.train.Feature(int64_list=tf.train.Int64List(value=[class_label])),
							"img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))
						} ))
						converter.write(example.SerializeToString())
					else:
						continue

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-v", "--verbosity", action="count",
						help="show info in each directory")
	
	args = parser.parse_args()
	for idx, itr_path in enumerate(path_list):
		data_converter(itr_path, data_list[idx], args.verbosity)