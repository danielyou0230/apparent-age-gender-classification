import os
import tensorflow as tf 
from PIL import Image

# Parameters
path_list = ['../X_data', '../T_data']
data_list = ["train.tfrecords", "test.tfrecords"]
age = ['child', 'young', 'adult', 'elder']
gender = ['male', 'female']

#
def data_converter(path, tf_data):
	with tf.python_io.TFRecordWriter(tf_data) as converter
		for idx_age, itr_age in enumerate(age):
			for idx_gender, itr_gender in enumerate(gender):
				class_label = [idx_age + idx_gender * 4]
				current_path = "{:s}/{:s}/{:s}".format(path, itr_age, itr_gender)
	
				for itr_file in os.listdir(current_path):
					if file.endswith('.jpg'):
						img_path = current_path + itr_file
						img = Image.open(img_path)
						# convert to tf format
						dic_label = tf.train.Int64List(value=[class_label])
						dic_data  = tf.train.BytesList(value=[img.tobytes()])
						# stream data to the converter
						example = tf.train.Example(features=tf.train.Features(
						feature=
						{ 
							"label"  : tf.train.Feature(int64_list=dic_label),
							"img_raw": tf.train.Feature(bytes_list=dic_data)
						} ))
						converter.write(example.SerializeToString())
					else:
						continue

if __name__ == '__main__':
	for idx, itr_path in enumerate(path_list):
		data_converter(itr_path, data_list[idx])
