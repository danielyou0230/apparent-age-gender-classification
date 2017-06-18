import tensorflow as tf
import argparse
import os 
from PIL import Image
import Modules as util

def demo_data_converter(path, tf_data):
	numlist = []
	buff = []
	with tf.python_io.TFRecordWriter(tf_data) as converter:
		n_file = sum(os.path.isfile(os.path.join(path, itr_dir)) \
							for itr_dir in os.listdir(path))
		print "{:s}: {:4d} files".format(path, n_file)
		for itr_file in os.listdir(path):
			print "{:s}: {:4d} files".format(path, n_file)
			if itr_file.endswith('.jpg'):
				img_path = path + itr_file
				img = Image.open(img_path, 0)
				img = img.resize((128, 128))
				img_raw = img.tobytes()
				# stream data to the converter
				example = tf.train.Example(features=tf.train.Features(
				feature=
				{ 
					"label"  : _int64_feature(0),
					"img_raw": _bytes_feature(img_raw)
				} ))

def run_model(args):
	print "hi"
	return
	# Extract faces
	util.face_extraction(args.demo_path)
	# Convert to tfrecord
	path = "{:s}_faces/".format(args.demo_path[:-1] if args.demo_path.endswith('/') \
								else args.demo_path)
	demo_data_converter(path, "demo.tfrecords")
	# Load demo data
	img, label = read_and_decode("demo.tfrecords")
	batch_img, batch_label = tf.train.shuffle_batch([img, label],
												batch_size=1000, capacity=1000,
												min_after_dequeue=300,
												allow_smaller_final_batch=True)
	# CNN Layers and attributes

	# Restore tensorflow session
	saver = tf.train.Saver()
	
	with tf.Session() as sess:
		# Restore variables from disk.
		saver.restore(sess, args.model)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("model", help="Path to the model file. (*.ckpt)")
	parser.add_argument("demo_path", help="Path to the demo files.")
	args = parser.parse_args()

	if not os.demo_path.isdir(args.demo_path):
		print "Demo path not found."

	else:
		if not os.path.isfile(args.model):
			print "No such file: {:s}".format(args.model)
		elif args.model.endswith(".ckpt"):
			run_model(args)
		else:
			print "Invalid file extension or wrong file."
