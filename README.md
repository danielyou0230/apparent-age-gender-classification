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

LBP by sk-image  
http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html  

## About the codes
ML_Final_Project.py: Top module for the project.  
data_conversion.py : Convert images to binary data for faster access for TensorFlow.  
Modules.py         : Modules used throughout the project.  
pending_code.py    : Codes that might be added to the project.  

## Roadmap
(Done) Facial Recognition  
(Done) Data Augmentation  
(Done) Convert to TensorFlow data type  
(Done) Implementing CNN with TensorFlow  
(Almost Done) Make the model usable  
(In Progress) Further tuning the model (Refine the model)  
(In Progress) Choice of optimizers and adaptive learning rate if valid  
(Coming Soon) Other Image Processing (LBP -> HOG -> ...)  
(----) Re-train the model and fine tuning  
(----) Demo code preparation  
(----) Report  

## Current Progress
### Working on...
1. Trying to converge the model faster.  
2. Other optimizers?  
3. Structure of the model  

### Issues
\#1 Load testing data as batch for validation.  
\#2 Unusable model (Cause UNKNOWN)  
\#3 Out-of-Memory... perhaps not well optimized :(

### TO-DOs
1. Parameterized the model  
2. Reduce training time and computational resources  
3. Better face recognition  
4. Find effective data augmentation methods  
5. [DONE] Increase the amount of data (size of dataset = 39xxx)  
6. Other image processing techniques  
7. Demo code preparation  
8. Adaptive learning rate  
9. Choose proper optimizer  
https://www.tensorflow.org/api_guides/python/train#Optimizers  

## Small scripts to make our life much easier  
### 1. routine  
Simple git commit and push at once, and add files if given.  
### How-to:  
#### Method 1. Run the file as bash  
> sh routine  

#### Method 2. Run the script as executable  
(required step, run the command just once when moving the file across machines)  
> chomod 777 routine  

(execute step)  
> ./routine  

### Usage:  
1. Just update files that already exist.  
Commit message: "routinely commit"
> ./routine  

2. Add one or multiple new files to the project.  
Commit message: "File(s): [list\_of\_files] added to the project"  
> ./routine [list\_of\_files]  

example:  
> ./routine file1 file2  

## 2. visualize  
Open Tensorboard (path adapted to our library path, change it if needed) without 
TensorBoard install by pip  
### Usage  
> ./visualize  

If TensorBoard is installed  
> tensorboard --logdir=/path/to/log-directory  

e.g. if the log stores in log/  
> tensorboard --logdir=log/  

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
> &nbsp;&nbsp;&nbsp;&nbsp;sess.run(init)  
> &nbsp;&nbsp;&nbsp;&nbsp;threads = tf.train.start_queue_runners(sess=sess)  
> &nbsp;&nbsp;&nbsp;&nbsp;minibatch = sess.run([data, label])   

Release GPU memory after computation  
https://github.com/tensorflow/tensorflow/issues/1578  
> config = tf.ConfigProto()  
> config.gpu_options.allow_growth=True  
> with tf.Session() as sess:  
> &nbsp;&nbsp;&nbsp;&nbsp;sess = tf.Session(config=config)  

Open TensorBoard on local machine from remote server  
https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server  
  
Login the remote server using command  
> ssh -L local_machine_port:127.0.0.1:6006 username@remote_server  

Where 127.0.0.1:6006 represents the port 6006 on remote_server, this 
command forwards all contents on the remote_server:6006 to your local machine.  
  
e.g. if 16006 is the port that we want to use as local_machine_port, then  
> ssh -L 16006:127.0.0.1:6006 username@remote_server  
  
To open Tensorboard, we can launch the TensorBoard on the remote server and 
simply accesss the port 16006 on our local machine.  
> 127.0.0.1:16006  
