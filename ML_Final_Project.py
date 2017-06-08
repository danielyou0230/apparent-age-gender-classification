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

numClass = 8

##################################################################
def read_and_decode(filename, img_size, depth):
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

		return img, label
read_and_decode('test.tfrecords')
##################################################################

# Parameters
learning_rate = 0.001
training_iters = 400000
batch_size = 128
display_step = 10
 
# Network Parameters
n_input = 10000 
n_classes = 8
dropout = 0.80 
 
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
 
# Create model
def conv2d(img, w, b):
	return tf.nn.relu(tf.nn.bias_add\
					  (tf.nn.conv2d(img, w,\
									strides=[1, 1, 1, 1],\
									padding='SAME'),b))
 
def max_pool(img, k):
	return tf.nn.max_pool(img, \
						  ksize=[1, k, k, 1],\
						  strides=[1, k, k, 1],\
						  padding='SAME')
 
# Store layers weight & bias
 # 5x5 conv, 1 input, 32 outputs
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32])) 
# 5x5 conv, 32 inputs, 64 outputs
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64])) 
# fully connected, 7*7*64 inputs, 1024 outputs
wd1 = tf.Variable(tf.random_normal([7*7*64, 1024])) 
# 1024 inputs, 10 outputs (class prediction)
wout = tf.Variable(tf.random_normal([1024, n_classes])) 
 
bc1 = tf.Variable(tf.random_normal([32]))
bc2 = tf.Variable(tf.random_normal([64]))
bd1 = tf.Variable(tf.random_normal([1024]))
bout = tf.Variable(tf.random_normal([n_classes]))
 
# Construct model
_X = tf.reshape(x, shape=[-1, 28, 28, 1])
 
# Convolution Layer
conv1 = conv2d(_X,wc1,bc1)
# Max Pooling (down-sampling)
conv1 = max_pool(conv1, k=2)
# Apply Dropout
conv1 = tf.nn.dropout(conv1,keep_prob)
 
# Convolution Layer
conv2 = conv2d(conv1,wc2,bc2)
# Max Pooling (down-sampling)
conv2 = max_pool(conv2, k=2)
# Apply Dropout
conv2 = tf.nn.dropout(conv2, keep_prob)
 
# Fully connected layer
# Reshape conv2 output to fit dense layer input
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]]) 
# Relu activation
dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1),bd1)) 
# Apply Dropout
dense1 = tf.nn.dropout(dense1, keep_prob) 
 
# Output, class prediction
pred = tf.add(tf.matmul(dense1, wout), bout)
 
#pred = conv_net(x, weights, biases, keep_prob)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
 
# Initializing the variables
#init = tf.initialize_all_variables()
init =tf.global_variables_initializer()
# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)########
		# Fit training using batch data
		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
		if step % display_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}"
				   .format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
		step += 1
	print ("Optimization Finished!")
	# Calculate accuracy for 256 mnist test images
	print ("Testing Accuracy:", 
		   sess.run(accuracy, feed_dict={x: mnist.test.images[:1024], 
										 y: mnist.test.labels[:1024], keep_prob: 1.}))


