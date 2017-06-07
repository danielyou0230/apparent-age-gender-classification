import os
import numpy as np
import ML_Final_Project_Modules as ml
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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
# Preprocessing
#ml.face_landmark_Preliminary()
#ml.export2csv(blur=True, sigma=2.0, hflip=True, vflip=True, \
#			  hvsplit=True, img_size=100, sample_size=100)

# Load Training data and target
##training_data = ml.load_data(file_training_normal)
##aug_list = ml.get_augment_list('CSV_Data/')
##for itr in aug_list:
##	training_data = np.vstack([training_data, ml.load_data(itr)])
##
##training_targ = ml.load_data(file_training_targ, ravel=True)
##
#### Load Testing data and target
##test_data = ml.load_data(file_testing_data)
##test_targ = ml.load_data(file_testing_targ, ravel=True)
##
### Use Neural Network as Feature Extractor
##neural_network = MLPClassifier(hidden_layer_sizes = (numfeatures, 100, ), \
##							   activation = 'relu', \
##							   solver = 'adam', \
##							   batch_size = 'auto', \
##							   learning_rate = 'adaptive', \
##							   max_iter = 200, \
##							   shuffle = True, \
##							   verbose = True)
##neural_network.fit(training_data, training_targ)
##extractor = neural_network.coefs_[0]
##train_features = training_data.dot(extractor)
##
##randomForest = RandomForestClassifier(n_estimators = numTrees, \
##									  min_samples_leaf = minLeafNode)
##randomForest = randomForest.fit(train_features, training_targ)
###test_features = test_data.dot(extractor)
##test_features = training_data.dot(extractor)
##prediction = randomForest.predict(test_features)
##
###ml.evaluate_result(prediction, test_targ, numClass)
##ml.evaluate_result(prediction, training_targ, numClass)


