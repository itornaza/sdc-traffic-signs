#** Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
###Writeup / README

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set mainly using the .shape method:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (Height, Width, Channels) = (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of sample images that correspond to each of the traffic signs. We can conclude that there is great variability on the number of samples between the different traffic signs ranging from the order of 100 to 2000 samples.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because we ca get their classification by only looking at their shapes

As a following step I applied histogram equalization to better distribute the pixel color values through out the [0, 255] range.

As a last step, I normalized the image data because it is a good technique to have zero variance and zero mean. I have chosen to normalize the grayscale channel values from [0, 255] to [-0.5, 0.5]

Here is an example of a traffic sign image before and after preprocessing

![alt text][image2]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers based on the lenet 5 model. My modification is the addition of dropout layers after the activation functions of the fully connected layers. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale and preprocessed images     | 
| Convolution           | 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU					|                                               |
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 3x3	    | 1x1 stride, valid padding, output 10x10x16	|
| ReLU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Flatten               | Outputs 400      								|	
| Fully connected		| Outputs 120        							|
| ReLU                  |                                               |
| Dropout               | Keep prob 0.65                                |
| Fully connected		| Outputs 84     								|
| ReLU                  |                                               |
| Dropout               | Keep prob 0.65                                |
| Fully connected		| Outputs 43     								| 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Tensorflow adamOptimizer. I have trained the network on my mac book pro 2,5 GHz Intel Core i7 processor and took approximatelly 6.034 minutes. I used a batch size of 128 since I did not encounter any delays from my cpu. I trained the network for 30 epochs with a learning rate of 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.954  
* test set accuracy of 0.940

An iterative approach was chosen:
* The first architecture that has been tested was the LeNet-5 architecture.
* The problem with this architecture was that the max validation accuracy was 0.93.
* The architecture was adjusted so a dropout layer was added after the activations of the first and second fullt connected layers in order to reduce the overfitting issue.
* The keep probability of the dropout layers was tested for settings varying from 0.5 to 0.7, and a design decision was made to go with a keep probability of 0.65 that achieved the best validation accuracy from the other values.
* The final model's accuracy on the training, validation and test set provide evidence that the model is working well above the 0.93 project requirement.
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because the color contrast between the white, gray and yellow areas is not so grate when converting to grayscale.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Priority road         | Priority road									|
| General caution       | General caution								|
| Speed limit 60 km/h	| Speed limit 60 km/h			 				|
| No passing			| No passing                                    |


The model was able to correctly guess 5 out of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.94

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14, 15, and 16 cells of the Ipython notebook.

For the first, third, forth and fifth image, the model is definately sure that this is a stop sign (probabilities of 0.998 to 0.999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998         			| Stop sign   									| 
| .744     				| Priority road  								|
| .999					| General caution								|
| .999	      			| Speed limit 60 km/h			 				|
| .999				    | No passing                                    |


For the second image the softmax probability is 0.744 which is far less than the other four.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


