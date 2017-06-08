# Apparent Age Gender Classification

## Course
Machine Learning @ NCTU EE (grad) 2016 Sep. - 2017 June

## About the Project (from TA's document)
In this project, you are asked to classify the input photos into
4 age groups and 2 genders. You are free to use any methods, tools, 
and language to finish this work.  
If you want to use any work that is not built by you, be sure to 
specify it in your report. Also, just repeating other people’s work 
won’t help you get high score in this subject, be sure to include 
your own idea in this work.  
  
A. Input:  
The input will be an RGB photo or Gray level photo with different 
size. The photos above are from the class child, young, adult, and 
elder, respectively.  
You are only allowed to use the photos in this dataset. (Data 
augmentation is permitted) We may randomly pick some team’ s work to 
retrain it to check if the result is far differed from your submission.  

B. Target  
For each photos, please classify it into one of the following classes. 
You can predict age and gender jointly or separately, but the final 
result need to be one of the classes above.  
Class is defined as follows:  
0: Male Child, 4: Female Child  
1: Male Young, 5: Female Young  
2: Male Adult, 6: Female Adult  
3: Male Elder, 7: Female Elder  

## Contributor
Daniel You, Vivian Chung

## Machine Learning Library Involved
Tensorflow, Dlib, scikit-learn

## Reference
Data Conversion  
http://blog.csdn.net/u012759136/article/details/52232266  
Tensorflow  
https://www.tensorflow.org/tutorials/layers  
MNIST using TensorFlow  
http://terrence.logdown.com/posts/1240896-play-tensorflow-mnist-handwriting-recognition  
Face Landmarks with dlib, Opencv, and Python  
http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/  
Dlib Intro. [Chinese]  
https://chtseng.wordpress.com/2016/12/23/dlib-%E5%A5%BD%E7%94%A8%E7%9A%84%E7%9A%84machine-learning%E5%B7%A5%E5%85%B7-%E4%B8%80/  
CNN example reference  
http://arbu00.blogspot.tw/2017/03/2-tensorflowconvolutional-neural.html  


## Function
ML_Final_Project.py: Top module for the project.  
data_conversion.py: Convert images to binary data for faster access for TensorFlow.  
Modules.py: Modules used throughout the project.  

## Roadmap
(Done) Facial Recognition  
(Done) Data Augmentation  
(Done) Convert to TensorFlow data type  
(Done) Implementing CNN with TensorFlow  
(In Progress) Make the model usable  
(----) Further tuning the model (Refine the model)  
(----) Report  

## Current Progress
### Working on...
Make the model usable... :(  

### Issues
\#1 Load testing data as batch for validation.  
\#2 Unusable model (Cause UNKNOWN)

### TO-DOs
1. Parameterized the model  
2. Reduce training time and computational resources  
3. Better face recognition  
4. Find effective data augmentation methods  
5. Increase the amount of data  

## Installation Reference
Dlib  
http://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/  
OpenCV  
http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/  
TensorFlow (build from source)  
https://www.tensorflow.org/install/install_sources  

## Reference for coding
Convert label to one-hot  
https://agray3.github.io/2016/11/29/Demystifying-Data-Input-to-TensorFlow-for-Deep-Learning.html  
> label = tf.stack(tf.one_hot(label-1, nClass))  

Fetch mini-batch from queue  
https://stackoverflow.com/questions/41978221/tensorflow-next-batch-of-data-from-tf-train-shuffle-batch  
> init = tf.global_variables_initializer()  
> with tf.Session() as sess:  
>&nbsp;&nbsp;&nbsp;&nbsp;sess.run(init)  
>&nbsp;&nbsp;&nbsp;&nbsp;threads = tf.train.start_queue_runners(sess=sess)  
>&nbsp;&nbsp;&nbsp;&nbsp;minibatch = sess.run([data, label])  
