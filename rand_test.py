import random
import os
import numpy as np
import pandas as pd
import shutil
import argparse

sample_size = 100
mainPath   = '../Dataset'
outputPath = '../Testing'
age = ['child', 'young', 'adult', 'elder']
gender = ['male', 'female']

def get_dataInfo(filepath):
	#print [itr_dir for itr_dir in os.listdir('Dataset/adult')]
	return sum(os.path.isfile(os.path.join(filepath, itr_dir)) \
							for itr_dir in os.listdir(filepath))

def cleanup_workspace(path2remove, ext='.jpg'):
	file_list = os.listdir(path2remove)
	for item in file_list:
		if item.endswith(ext):
			os.remove(os.path.join(path2remove, item))

def generate_index(path, sample_size):
	n_file = get_dataInfo(path)
	print "{:s}: {:d} files".format(path, n_file) 
	# Generate the index of the image and the corresponding augmentation mode
	sample_list = random.sample(xrange(n_file / 2), sample_size)
	return sample_list

def random_pick_pictures(sample_size=100):
	# Cleanup workspace or make directory
	target = list()
	cleanup_workspace(outputPath) if os.path.isdir(outputPath) else os.makedirs(outputPath)
	index = 0
	for idx_age, itr_age in enumerate(age):
		for idx_gen, itr_gen in enumerate(gender):
			token = idx_age + idx_gen * 4
			curr_path = "{:s}/{:s}/{:s}".format(mainPath, itr_age, itr_gen)
			index_list = generate_index(curr_path, sample_size)
			for itr in index_list:
				file = "{:s}/{:d}.jpg".format(curr_path, itr)
				newfile = "{:s}/{:d}.jpg".format(outputPath, index)
				shutil.copy(file, newfile)
				target.append(token)
				index += 1
	df = pd.DataFrame(target)
	df.to_csv(outputPath+'/T.csv', header=False, index=False)
	print "{:d} files picked for self-verifying".format(index)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("n_test", type=int, help="test amount for each class")
	args = parser.parse_args()
	
	if not os.path.isdir(outputPath):
		os.makedirs(outputPath)
	
	random_pick_pictures(args.n_test)
