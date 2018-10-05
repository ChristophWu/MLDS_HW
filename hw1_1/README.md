# README

## How to implement and get the results
* python HW1_1_1.py to see the loss and simulation curve of Q1.
* python HW1_1_2.py to see the accuracy and loss curve of Q2.

## Introdcution
This work aims to find the relationship between the depth of a model and the power of a model. And I will do two tasks to figure out it.
![](https://i.imgur.com/RyYts1a.png)


## Simulate a Function
### The models I use
* ![](https://i.imgur.com/EaMw6S8.png)
* Three models with the same number of parameters(571 weights) are introduced.
* Only the depth of these models are different.

### The function I use
* function1: y = np.sin(5 * np.pi * x)/(5 * np.pi * x)
* function2: y = np.sin(2 * np.pi * x * x)/(2 * np.pi * x)

### The training loss of all models
function1
![](https://i.imgur.com/D2mhtjx.png)
function2
![](https://i.imgur.com/8pS1xTA.png)

### The predicted function curve of all models
function1
![](https://i.imgur.com/wQvYutq.png)
function2
![](https://i.imgur.com/smpXXBM.png)

### Conclusion
* The more deeper, the less loss.
* The more deeper, the more accuracy can a model simulate a function. 

## Train on Actual Tasks
### The models I use
* ![](https://i.imgur.com/oQwdcIt.png)
The deep one is consist of two convolution layers, two maxpooling layers and one fully connected layer with 8480 parameters.


* ![](https://i.imgur.com/58ppvYo.png)
The shallow one is consist of one convolution layer, one maxpooling layer and one fully connected layer with 9925 parameters.

* I apply these two models to mnist dataset.

### Training loss of all models
![](https://i.imgur.com/67HDHOf.png)

### Training accuracy of all models
![](https://i.imgur.com/D3XW5jB.png)

### Conclusion
The deeper model can reach lower loss and can predict much more accurate.

