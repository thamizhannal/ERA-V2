## Assignment Part A

1. Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github. 
   1. **Use exactly the same values for all variables as used in the class**
   2. Take a screenshot, and show that screenshot in the readme file
   3. Excel file must be there for us to cross-check the image shown on readme (no image = no score)
   4. Explain each major step
   5. Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 



### Example Neural Network

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/NeuralNetwork.png)

### Forward Pass:

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/forwardpass.png)

### Back Propagation:

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/BackPropGradComp.png)

### Error Values for various Learning rates
These are error values against various learning rates, as the epoch progresses errors start reducing.

![alt text](https://raw.githubusercontent.com/thamizhannal/END3.0/main/Session%202%20-%20Backprop%2C%20embeddings%20and%20Language%20Models/images/ErrorTable.png)



# Assignment Part B
# MINIST Classifier in 20k parameter with 99.4% validation Accuracy

### Objective: ###

Refer to this code: COLABLINK https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx
WRITE IT AGAIN SUCH THAT IT ACHIEVES
- 99.4% validation accuracy
- Less than 20k Parameters
- You can use anything from above you want.
- Less than 20 Epochs
- No fully connected layer
- To learn how to add different things we covered in this session, you can refer to this code: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99 DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.

####  Approach & Architecture:   ###

**Input Data Set:** MNIST dataset consists of human hand-written monochrome images of size 28x28. Our objective is to detect human 
handwritten digits using simple DNN.

**Network Architecture:** The architecture chosen was squeeze & expansion 
architecture to detect numbers in MINIST dataset. This architecture consist of convolution blocks followed by transition blocks.

**Convolution Block:** Each convolution block consist of two convolution layer with kernel size 3x3
followed by Max pooling (size 2x2). After every convolution layer Batch Normalization was used and 
before end of convolution block Drop out regularization was used. <r>
Conv1 Block:
At the end of the first Convolution block, Receptive Field size is 10x10, in this stage the 
edges & gradients and texture & pattern could have been detected.

**Edges and Gradients:** Since input images are small in size, our network can expected extract edges and gradients at the Receptive Fields of 5-7.
It is required to have 2-3 convolution layers to detect edges & gradients

**Max Pooling:** It filters out the least important features and sends out most important features 
to consecutive layers for prediction.

**Batch Normalization** The batch normalization after every convolution(cond2d()). BN 
helped us to standardize the input to a convolution layer for every mini-batch. This stabilizes the learning process and also accelerates the DNN training.

**Adaptive Avg Pooling** AdaptiveAvgPool2d in before prediction that calculates the average value for each pixel on the feature map.

**Train Vs Validation Plot**  The train loss, train accuracy, validation loss & 
validation accuracies has been captured and printed for reference. 


### **Conclusion:**
A simple convolution neural network architecture, that consist of two convolution blocks and 
transition blocks (CT) with default batch size=32, epoch=20 and optimizer=SGD, has been 
implemented using 19,046 parameters and achieved validation accuracy of 99.41% in 11th epoch.
