# Apparent Age Gender Classification

## Course
Machine Learning @ NCTU EE (grad) 2016 Sep. - 2017 June

## About the Project (from TA's document)
In this project, you are asked to classify the input photos into 4 age groups and 2 genders. You are free to use any methods, tools, and language to finish this work. <br/>
If you want to use any work that is not built by you, be sure to specify it in your report. Also, just repeating other people’s work won’t help you get high score in this subject, be sure to include your own idea in this work. <br/> <br/>
A. Input: <br/>
The input will be an RGB photo or Gray level photo with different size. The photos above are from the class child, young, adult, and elder, respectively. <br/>
You are only allowed to use the photos in this dataset. (Data augmentation is permitted) We may randomly pick some team’ s work to retrain it to check if the result is far differed from your submission. <br/>

B. Target <br/>
For each photos, please classify it into one of the following classes. You can predict age and gender jointly or separately, but the final result need to be one of the classes above. <br/>
Class is defined as follows: <br/>
0: Male Child, 4: Female Child <br/>
1: Male Young, 5: Female Young <br/>
2: Male Adult, 6: Female Adult <br/>
3: Male Elder, 7: Female Elder <br/>

## Contributor
Daniel You, Vivian Chung

## Machine Learning Library Involved
Tensorflow, Dlib, scikit-learn

## Reference
Data Conversion <br/> 
http://blog.csdn.net/u012759136/article/details/52232266 <br/>
Tensorflow      <br/>
https://www.tensorflow.org/tutorials/layers <br/>
MNIST using TensorFlow <br/>
http://terrence.logdown.com/posts/1240896-play-tensorflow-mnist-handwriting-recognition <br/>
Face Landmarks with dlib, Opencv, and Python <br/>
http://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ <br/>


## Function
ML\_Final\_Project.py: Top module for the project. <br/>
data\_conversion.py: Convert images to binary data for faster access for TensorFlow.<br />
Modules.py: Modules used throughout the project. <br/>

## Roadmap
(Done) Facial Recognition <br/>
(Done) Data Augmentation  <br/>
(Done) Convert to TensorFlow data type <br/>
(In Progress) Implementing CNN with TensorFlow <br/>
(----) Tuning model <br/>
(----) Report <br/>

## Installation Reference
Dlib <br/>
http://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/ <br/>
