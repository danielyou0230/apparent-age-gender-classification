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
	batch_size = 50
	display_step = 10
	
	# Network Parameters
	n_input = pow(image_size, 2)
	dropout = 0.8
	## Layer parameters
	kernel_units = [depth, 64, 128, 256, 512, 1024, 2048]
	kernel = [16, 16]
	# Fully connected inputs
	#pool_factor = pow(2, len(kernel_size))
	pool_factor = pow(2, 5)
	n_connected = pow(image_size / pool_factor, 2)
	
	# tf Graph input
	#x = tf.placeholder(tf.float32, [None, n_input])
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
	#pool1 = tf.nn.dropout(pool1, keep_prob) 
	
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
	#pool2 = tf.nn.dropout(pool2, keep_prob) 
	
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
	#pool3 = tf.nn.dropout(pool3, keep_prob) 
	
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
	#pool4 = tf.nn.dropout(pool4, keep_prob) 
	
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
	#pool5 = tf.nn.dropout(pool5, keep_prob) 
	
	# Dense Layer
	flatten = tf.reshape(pool5, [-1, n_connected * kernel_units[5]]) 
	dense = tf.layers.dense(
			inputs=flatten, 
			units=kernel_units[6], 
			activation=tf.nn.relu,
			name='dense' )
	dense = tf.nn.dropout(dense, keep_prob) 
	#fc1 = tf.contrib.layers.fully_connected(
	#	  inputs=pool5, 
	#	  num_outputs=kernel_units[6], 
	#	  activation_fn=tf.nn.relu )
	#
	#pred = tf.contrib.layers.fully_connected(
	#	  inputs=fc1, 
	#	  num_outputs=n_classes, 
	#	  activation_fn=tf.nn.relu )
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
													min_after_dequeue=300,
													allow_smaller_final_batch=True)
	# Load testing data
	t_img, t_label = read_and_decode(file[1])
	test_img, test_lbl = tf.train.shuffle_batch([t_img, t_label],
												 batch_size=800, capacity=800,
												 min_after_dequeue=10,
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
		sess = tf.Session(config=config)
		writer = tf.summary.FileWriter('board/', graph=sess.graph)
		sess.run(init)
		threads = tf.train.start_queue_runners(sess=sess)
	
		step = 1
		prev_loss = 0.
		stagnant = 0
		# Keep training until reach max iterations
		#while step <= 10:
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
				#tf.summary.scalar("Loss:", loss)
				print "Iter {:6d}, Minibatch Loss = {:.6f}, Training Accuracy = {:2.2f}%" \
					  .format(step * batch_size, loss, acc * 100)
				#  .format(step * batch_size, loss, acc * 100)
				delta_loss = prev_loss - loss
				#if acc > 0.95 and abs(delta_loss) < 0.001 and step > 100:
				#	stagnant += 1
				#	if stagnant == 2:
				#		print "Two consecutive losses change < 0.01, stopping..."
				#		break
				#else:
				#	prev_loss = loss
				#	stagnant = 0

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
		# 
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--lbp", action="store_true", help="Use LBP data")
	parser.add_argument("-n", "--normal", action="store_true", help="Use normal data")

	args = parser.parse_args()
	run_model(args)