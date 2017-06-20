import os
import numpy as np
import Modules as ml
import tensorflow as tf
import argparse

norm = ["tfrecords/train.tfrecords", "tfrecords/test.tfrecords"]
lbp  = ["tfrecords/train_lbp.tfrecords", "tfrecords/test_lbp.tfrecords"]

n_classes = 8
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
# Parameters
##################################################################
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
#read_and_decode('test.tfrecords')

def load_tfrecord_batch(filename):
	for serialized_example in tf.python_io.tf_record_iterator(filename):
		example = tf.train.Example()
		example.ParseFromString(serialized_example)
		image = example.features.feature['img_raw'].bytes_list.value
		label = example.features.feature['label'].int64_list.value
		print image
		print label

##################################################################
def run_model(args):
	file = lbp if args.lbp else norm
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
	
	# Class Prediction (output layer)
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
	#flatten = tf.reshape(pool4, [-1, n_connected * 4 * kernel_units[4]]) 
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
	
	# Load training data
	img, label = read_and_decode(file[0])
	batch_img, batch_label = tf.train.shuffle_batch([img, label],
													batch_size=batch_size, capacity=1000,
													min_after_dequeue=100,
													allow_smaller_final_batch=True)
	# Load testing data
	t_img, t_label = read_and_decode(file[1])
	test_img, test_lbl = tf.train.shuffle_batch([t_img, t_label],
												 batch_size=800, capacity=800,
												 min_after_dequeue=0,
												 allow_smaller_final_batch=True)
	
	# TensorBoard
	tf.summary.scalar("Accuracy:", accuracy)
	merged = tf.summary.merge_all()
	
	# Saver
	saver = tf.train.Saver()
	# Initializing the variables
	init = tf.global_variables_initializer()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True

	# Launch the graph
	with tf.Session() as sess:
		#sess = tf.Session(config=config)
		writer = tf.summary.FileWriter('board/', graph=sess.graph)
		init = tf.global_variables_initializer()
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
		step = 1
		prev_loss = 0.
		stagnant = 0
		# Keep training until reach max iterations
		while step * batch_size <= training_iters:
			batch_xs, batch_ys = sess.run([batch_img, batch_label])
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
			if step % display_step == 0:
				# Calculate batch accuracy
				tfb_summary, acc = sess.run([merged, accuracy], \
											  feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
				writer.add_summary(tfb_summary, step * batch_size)
				# Calculate batch loss
				loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
				#writer.add_summary(tfb_summary, step * batch_size)
				print "Iter {:6d}, Minibatch Loss = {:.6f}, Training Accuracy = {:2.2f}%" \
					  .format(step * batch_size, loss, acc * 100)
				delta_loss = prev_loss - loss
				if step * batch_size % 10000 == 0:
					batch_tx, batch_ly = sess.run([test_img, test_lbl])
					pred_class = sess.run(tf.argmax(pred, 1), \
										  feed_dict={x: batch_tx, y: batch_ly, keep_prob: 1.})
					pred_class = list(pred_class)
					print "0: {:d}, 1: {:d}, 2: {:d}, 3: {:d}, 4: {:d}, 5: {:d}, 6: {:d}, 7: {:d}" \
								  .format(pred_class.count(0), pred_class.count(1), \
										  pred_class.count(2), pred_class.count(3), \
										  pred_class.count(4), pred_class.count(5), \
										  pred_class.count(6), pred_class.count(7) )
					validation_acc = sess.run(accuracy, \
									 feed_dict={x: batch_tx, y: batch_ly, keep_prob: 1.})
					print "Testing Accuracy: {:.3f}%".format(validation_acc * 100.)
			step += 1
		print "Optimization Finished!"
		batch_tx, batch_ly = sess.run([test_img, test_lbl])
		pred_class = sess.run(tf.argmax(pred, 1), \
							  feed_dict={x: batch_tx, y: batch_ly, keep_prob: 1.})
		pred_class = list(pred_class)
		print "0: {:d}, 1: {:d}, 2: {:d}, 3: {:d}, 4: {:d}, 5: {:d}, 6: {:d}, 7: {:d}" \
					  .format(pred_class.count(0), pred_class.count(1), \
							  pred_class.count(2), pred_class.count(3), \
							  pred_class.count(4), pred_class.count(5), \
							  pred_class.count(6), pred_class.count(7) )
		validation_acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ly, keep_prob: 1.})
		print "Testing Accuracy: {:.3f}%".format(validation_acc * 100.)
		# Save model 
		save_ckpt = saver.save(sess, "model.ckpt")
		print "Model saved in file: {:s}".format(save_ckpt)
		coord.request_stop()
		coord.join(threads)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--lbp", action="store_true", help="Use LBP data")
	parser.add_argument("-n", "--normal", action="store_true", help="Use normal data")
	
	args = parser.parse_args()
	run_model(args)
