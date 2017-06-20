import tensorflow as tf
import argparse
import os 
from PIL import Image
import Modules as util
import random
import pandas as pd
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

n_classes = 8

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def demo_data_converter(path, tf_data, selfeval):
	numlist = []
	buff = []
	idx = 0
	if selfeval:
		raw_target = pd.read_csv('Testing/T.csv', header=None)
		target = raw_target.as_matrix()
		target = list(target.ravel())

	itr = 0
	with tf.python_io.TFRecordWriter(tf_data) as converter:
		n_file = sum(os.path.isfile(os.path.join(path, itr_dir)) \
							for itr_dir in os.listdir(path))
		all_files = os.listdir(path)
		all_files.sort(key=lambda f: int(filter(str.isdigit, f)))
		for itr_file in all_files:
			if itr_file.endswith('.jpg'):
				img_path = path + itr_file
				img = Image.open(img_path)
				img = img.resize((128, 128))
				img_raw = img.tobytes()
				# stream data to the converter
				example = tf.train.Example(features=tf.train.Features(
				feature=
				{ 
					"label"  : _int64_feature([target[idx]]) if selfeval else 
							   _int64_feature([1]),
					"img_raw": _bytes_feature(img_raw)
				} ))
				converter.write(example.SerializeToString())
				itr += 1
			else:
				continue
		print "{:s}: {:4d} files".format(path, itr)

def read_and_decode(filename, img_size=128, depth=1):
	if not filename.endswith('.tfrecords'):
		print "Invalid file \"{:s}\"".format(filename)
		return [], []
	else:
		data_queue = tf.train.string_input_producer([filename])

		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(data_queue) 
		features = tf.parse_single_example(serialized_example,
				   features={
							 'label'   : tf.FixedLenFeature([], tf.int64),
							 'img_raw' : tf.FixedLenFeature([], tf.string),
							})

		img = tf.decode_raw(features['img_raw'], tf.uint8)
		img = tf.reshape(img, [img_size, img_size, depth])
		# Normalize the image
		img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
		label = tf.cast(features['label'], tf.int32)
		label_onehot = tf.stack(tf.one_hot(label, n_classes))
		return img, label_onehot

def run_model(args):
	## Extract faces
	print "Detecting and extracting faces from images..."
	undetectLst, total = util.face_extraction(args.demo_path)

	## Convert to tfrecord
	path = "{:s}_faces/".format(args.demo_path[:-1] if args.demo_path.endswith('/') \
								else args.demo_path)

	print "Converting face images to tfrecord..."
	demo_data_converter(path, "demo.tfrecords", args.selfeval)

	print "Initializing model..."
	# CNN Layers and attributes
	# Data properties
	image_size = 128
	depth = 1
	# Parameters
	learning_rate = 0.007
	training_iters = 1000100
	batch_size = 25
	display_step = 10
	
	# Network Parameters
	n_input = pow(image_size, 2)
	dropout = 0.8
	## Layer parameters
	kernel_units = [depth, 64, 128, 256, 512, 1024, 2048]
	kernel = [16, 16]
	# Fully connected inputs
	pool_factor = pow(2, 5)
	n_connected = pow(image_size / pool_factor, 2)
	
	# tf Graph input
	x = tf.placeholder(tf.float32, [None, image_size, image_size, depth], name='input')
	y = tf.placeholder(tf.float32, [None, n_classes], name='label')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob') 
	
	# Class Prediction
	w_out   = tf.Variable(tf.random_normal([kernel_units[6], n_classes])) 
	
	bias_d_out = tf.Variable(tf.random_normal([n_classes]))
	
	# Convolution Layer
	conv1 = tf.layers.conv2d(
		  inputs=x,
		  filters=kernel_units[1],
		  kernel_size=[5, 5],
		  padding="same",
		  activation=tf.nn.relu,
		  use_bias=True,
		  kernel_initializer=None,
		  bias_initializer=tf.zeros_initializer(),
		  name='conv1' ) 
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
	
	conv2 = tf.layers.conv2d(
		  inputs=pool1,
		  filters=kernel_units[2],
		  kernel_size=[4, 4],
		  padding="same",
		  activation=tf.nn.relu,
		  use_bias=True,
		  kernel_initializer=None,
		  bias_initializer=tf.zeros_initializer() ,
		  name='conv2') 
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
	
	conv3 = tf.layers.conv2d(
		  inputs=pool2,
		  filters=kernel_units[3],
		  kernel_size=[3, 3],
		  padding="same",
		  activation=tf.nn.relu,
		  use_bias=True,
		  kernel_initializer=None,
		  bias_initializer=tf.zeros_initializer(),
		  name='conv3' ) 
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name='pool3')

	conv4 = tf.layers.conv2d(
		  inputs=pool3,
		  filters=kernel_units[4],
		  kernel_size=[2, 2],
		  padding="same",
		  activation=tf.nn.relu,
		  use_bias=True,
		  kernel_initializer=None,
		  bias_initializer=tf.zeros_initializer(),
		  name='conv4' ) 
	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, name='pool4')
	
	conv5 = tf.layers.conv2d(
		  inputs=pool4,
		  filters=kernel_units[5],
		  kernel_size=[2, 2],
		  padding="same",
		  activation=tf.nn.relu,
		  use_bias=True,
		  kernel_initializer=None,
		  bias_initializer=tf.zeros_initializer(),
		  name='conv5' ) 
	pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2, name='pool5')
	
	# Dense Layer
	flatten = tf.reshape(pool5, [-1, n_connected * kernel_units[5]]) 
	dense = tf.layers.dense(
			inputs=flatten, 
			units=kernel_units[6], 
			activation=tf.nn.relu,
			name='dense' )
	dense = tf.nn.dropout(dense, keep_prob) 

	# Output, class prediction
	pred = tf.add(tf.matmul(dense, w_out), bias_d_out)
	
	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
	optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate).minimize(cost)
	#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	print "Tensorflow session begin"
	pred_list = list()
	# Restore tensorflow session
	with tf.Session() as sess:
		# Restore variables from disk.
		print "Restoring model..."
		saver = tf.train.Saver()
		saver = tf.train.import_meta_graph(args.model)
		saver.restore(sess, tf.train.latest_checkpoint("./"))

		img, label = read_and_decode("demo.tfrecords")
		#img, label = read_and_decode("tfrecords/test.tfrecords")
		batch_img, batch_label = tf.train.batch([img, label],
												 batch_size=total, capacity=total,
												 allow_smaller_final_batch=True)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		print "Loading testing data..."
		batch_tx, batch_ly = sess.run([batch_img, batch_label])
		print "Predicting testing data..."
		pred_class = sess.run(tf.argmax(pred, 1), \
							  feed_dict={x: batch_tx, y: batch_ly, keep_prob: 1.})
		pred_list = list(pred_class)

		print "0: {:d}, 1: {:d}, 2: {:d}, 3: {:d}, 4: {:d}, 5: {:d}, 6: {:d}, 7: {:d}" \
					  .format(pred_list.count(0), pred_list.count(1), \
							  pred_list.count(2), pred_list.count(3), \
							  pred_list.count(4), pred_list.count(5), \
							  pred_list.count(6), pred_list.count(7) )
		coord.request_stop()
		coord.join(threads)

	# Generate file name list for output
	name_list = list()
	for itr in os.listdir(path):
		if itr.endswith('.jpg'):
			name_list.append(itr)

	# Self-evaluation mode
	if args.selfeval:
		print "Self-evaluating mode:"
		score = 0
		raw_target = pd.read_csv('Testing/T.csv', header=None)
		target = raw_target.as_matrix()
		target = list(target.ravel())
		for idx, itr in enumerate(target):
			if pred_list[idx] == itr:
				score += 1
		print "Accuracy: {:2.2f}%".format(100.0 * score / 800)

	print "Exporting predictions to CSV file..."
	# Export predictions and file names to CSV file
	name_list.sort(key=lambda f: int(filter(str.isdigit, f)))
	name_list = np.vstack(name_list)
	pred_list = np.vstack(pred_list)
	output_content = np.hstack([name_list, pred_list])
	df = pd.DataFrame(output_content)
	df.to_csv('Prediction.csv', header=False, index=False)
	print "Done. Output file \"Prediction.csv\"" 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("model", help="Path to the model file. (*.ckpt)")
	parser.add_argument("demo_path", help="Path to the demo files.")
	parser.add_argument("-s", "--selfeval", action='store_true',
						help="self evaluating mode")
	args = parser.parse_args()

	if not os.path.isdir(args.demo_path):
		print "Demo path not found."

	else:
		if not os.path.isfile(args.model):
			print "No such file: {:s}".format(args.model)
		elif args.model.endswith(".ckpt.meta"):
			run_model(args)
		else:
			print "Invalid file extension or wrong file."
