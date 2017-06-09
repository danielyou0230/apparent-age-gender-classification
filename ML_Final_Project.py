import os
import numpy as np
import Modules as ml
import tensorflow as tf

# Parameters
file_training_normal = "CSV_Data/training_normal.csv"
file_training_targ   = "CSV_Data/training_target.csv"

file_info          = "CSV_Data/processed_amount.csv"
file_testing_data  = "CSV_Data/testing_data.csv"
file_testing_targ  = "CSV_Data/testing_target.csv"

numfeatures = 500
numTrees = 1000
minLeafNode = 300
n_classes = 8
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
# Data properties
image_size = 128
depth = 1
# Parameters
learning_rate = 0.001
training_iters = 80001
batch_size = 50
display_step = 10

# Network Parameters
n_input = pow(image_size, 2)
dropout = 0.7
## Layer parameters
feature_map = [depth, 128, 64, 32]
kernel_size = [16, 16]
# Fully connected inputs
#pool_factor = pow(2, len(kernel_size))
pool_factor = pow(2, 5)
n_connected = pow(image_size / pool_factor, 2)

# tf Graph input
#x = tf.placeholder(tf.float32, [None, n_input])
x = tf.placeholder(tf.float32, [None, image_size, image_size, depth], name='input')
y = tf.placeholder(tf.float32, [None, n_classes], name='label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') 

# Create model
def conv2d_relu(img, w, b):
	return tf.nn.relu(tf.nn.bias_add\
					  (tf.nn.conv2d(img, w,\
									strides=[1, 1, 1, 1],\
									padding='SAME'), b))

def max_pool(img, k):
	return tf.nn.max_pool(img, \
						  ksize=[1, k, k, 1],\
						  strides=[1, k, k, 1],\
						  padding='SAME')

# Store layers weight & bias
# 5x5 conv, 1 input, 32 outputs (kernels)
#w_conv1 = tf.Variable(tf.random_normal([kernel_size[0], kernel_size[0], feature_map[0], 
#										feature_map[1]])) 
#
#w_conv2 = tf.Variable(tf.random_normal([kernel_size[1], kernel_size[1], feature_map[1], 
#										feature_map[2]])) 
#
#w_conv3 = tf.Variable(tf.random_normal([kernel_size[2], kernel_size[2], feature_map[2], 
#										feature_map[3]])) 
# Fully connected Layer
#w_dens1 = tf.Variable(tf.random_normal([n_connected * feature_map[2], 
#										feature_map[3]])) 
# Class Prediction)
#w_out   = tf.Variable(tf.random_normal([feature_map[3], n_classes])) 
w_out   = tf.Variable(tf.random_normal([2048, n_classes])) 

#bias_conv1 = tf.Variable(tf.random_normal([feature_map[1]]))
#bias_conv2 = tf.Variable(tf.random_normal([feature_map[2]]))
#bias_conv3 = tf.Variable(tf.random_normal([feature_map[3]]))
#bias_dens1 = tf.Variable(tf.random_normal([feature_map[3]]))
bias_d_out = tf.Variable(tf.random_normal([n_classes]))

# Construct model
#_X = tf.reshape(x, shape=[-1, image_size, image_size, depth])

# Convolution Layer
#conv1 = conv2d_relu(_X, w_conv1, bias_conv1)
#conv1 = conv2d_relu(x, w_conv1, bias_conv1)
## Max Pooling (down-sampling)
#conv1 = max_pool(conv1, k=2)
## Apply Dropout
#conv1 = tf.nn.dropout(conv1, keep_prob)

conv1 = tf.layers.conv2d(
	  inputs=x,
	  filters=64,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu,
	  use_bias=True,
	  kernel_initializer=None,
	  bias_initializer=tf.zeros_initializer(),
	  name='conv1' ) 

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='pool1')
pool1 = tf.nn.dropout(pool1, keep_prob) 

# Convolution Layer
#conv2 = conv2d_relu(conv1, w_conv2, bias_conv2)
## Max Pooling (down-sampling)
#conv2 = max_pool(conv2, k=2)
## Apply Dropout
#conv2 = tf.nn.dropout(conv2, keep_prob)

conv2 = tf.layers.conv2d(
	  inputs=pool1,
	  filters=128,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu,
	  use_bias=True,
	  kernel_initializer=None,
	  bias_initializer=tf.zeros_initializer() ,
	  name='conv2') 

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='pool2')
pool2 = tf.nn.dropout(pool2, keep_prob) 

# Convolution Layer
conv3 = tf.layers.conv2d(
	  inputs=pool2,
	  filters=256,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu,
	  use_bias=True,
	  kernel_initializer=None,
	  bias_initializer=tf.zeros_initializer(),
	  name='conv3' ) 

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name='pool3')
pool3 = tf.nn.dropout(pool3, keep_prob) 

# Convolution Layer
conv4 = tf.layers.conv2d(
	  inputs=pool3,
	  filters=512,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu,
	  use_bias=True,
	  kernel_initializer=None,
	  bias_initializer=tf.zeros_initializer(),
	  name='conv4' ) 

pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, name='pool4')
pool4 = tf.nn.dropout(pool4, keep_prob) 

# Convolution Layer
conv5 = tf.layers.conv2d(
	  inputs=pool4,
	  filters=1024,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu,
	  use_bias=True,
	  kernel_initializer=None,
	  bias_initializer=tf.zeros_initializer(),
	  name='conv5' ) 

pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2, name='pool5')
pool5 = tf.nn.dropout(pool5, keep_prob) 

# Dense Layer
#pool2_flat = tf.reshape(pool2, [-1, w_dens1.get_shape().as_list()[0]]) 
flatten = tf.reshape(pool5, [-1, n_connected * 1024]) 
dense = tf.layers.dense(
		inputs=flatten, 
		units=2048, 
		activation=tf.nn.relu,
		name='dense' )
dense = tf.nn.dropout(dense, keep_prob) 

# Relu activation
#dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, w_dens1), bias_dens1)) 
# Apply Dropout
#dense1 = tf.nn.dropout(dense1, keep_prob) 

# Output, class prediction
pred = tf.add(tf.matmul(dense, w_out), bias_d_out)

# Generate Predictions
#predictions = { 
#				"classes": tf.argmax(input=logits, axis=1),
#				"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
#			  }

#pred = conv_net(x, weights, biases, keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Load training data
img, label = read_and_decode("train.tfrecords")
batch_img, batch_label = tf.train.shuffle_batch([img, label],
												batch_size=batch_size, capacity=300,
												min_after_dequeue=200,
												allow_smaller_final_batch=True)
# Load testing data
t_img, t_label = read_and_decode("test.tfrecords")
test_img, test_lbl = tf.train.shuffle_batch([t_img, t_label],
											 batch_size=800, capacity=800,
											 min_after_dequeue=10,
											 allow_smaller_final_batch=True)
# Initializing the variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# Launch the graph
with tf.Session() as sess:
	sess = tf.Session(config=config)
	writer = tf.summary.FileWriter('tflog/', sess.graph)
	sess.run(init)
	threads = tf.train.start_queue_runners(sess=sess)
	#for i in range(3):
	#	val, l = sess.run([batch_img, batch_label])
	#	#l = to_categorical(l, 12) 
	#	print(val.shape, l)

	step = 1
	prev_loss = 0.
	stagnant = 0
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_xs, batch_ys = sess.run([batch_img, batch_label])
		#batch_xs, batch_ys = mnist.train.next_batch(batch_size)########
		# Fit training using batch data
		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
		if step % display_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			print "Iter {:7d}, Minibatch Loss = {:f}, Training Accuracy = {:2.2f}%" \
				  .format(step * batch_size, loss, acc * 100)
			
			#delta_loss = prev_loss - loss
			#if acc > 0.80 and abs(delta_loss) < 0.03 and step > 100:
			#	stagnant += 1
			#	if stagnant == 2:
			#		print "Two consecutive losses change < 0.03, stopping..."
			#		break
			#else:
			#	prev_loss = loss
			#	stagnant = 0
		step += 1
	print "Optimization Finished!"
	# 
	batch_tx, batch_ly = sess.run([test_img, test_lbl])
	#print "size of test{:s}".format(batch_tx.shape)
	print "Testing Accuracy:", \
		   sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ly, keep_prob: 1.})

