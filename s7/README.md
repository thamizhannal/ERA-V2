# S7 Assignment

### Objective: ###

MNIST Classification with 99.4% accuracy in less than 8k parameters <br>

Your new target is:
* 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
* Less than or equal to 15 Epochs
* Less than 8000 Parameters
* Do this using your modular code. Every model that you make must be there in the model.py file 
  as Model_1, Model_2, etc.
* Do this in exactly 3 steps

##  Net1 MNIST classifier   ##
![Net1.png](Net1.png)

## Target:
* Get the set-up right <br>
* Set Transforms <br>
* Set Data Loader <br>
* Set Basic Working Code <br>
* Set Basic Training  & Test Loop <br>
* Plot Train vs Test Loss graph <br>
## Results:
* Parameters: 6.3M <br>
* Best Training Accuracy: 99.95 <br>
* Best Test Accuracy: 99.32 <br>
## Analysis:
* Net1 is heavy model for simple MNIST digit classifier. <br>
* The train accuracy is reached almost 100% and test accuracy is still 99.25% and there is not much scope to learn from train data and improve the test accuracy. <br>
* Model is over-fitting, we need to create a light weight robust model in next iteration. <br>

##  Net2 MNIST classifier   ##

![Net2.png](Net2.png)
## Target:
* Create a light-weight model (Net2) from basic Model(Net1). <br>
* Reduce Kernel size in multi-fold to reduce the overall trainable parameters <br>
* Use Max Pooling and Adaptive Average Pooling <br>
## Results:
* Parameters: 21k <br>
* Best Training Accuracy: 99.77 <br>
* Best Test Accuracy: 99.34 <br>
## Analysis:
* Net2 is a robust light-weight model with 20k parameters that works well till ~99.2 to 99.3% of 
  test accuracy and after that train and test accuracy starts dropping.<br>
* The model starts over fitting after it reached the max train accuracy of 99.7% and no further 
  improvements is possible.<br>
* Need to apply regularization methods <br>
* 
##  Net3 MNIST classifier   ##
![Net3.png](Net3.png)
## Target:
* Create a light-weight model (Net3) with lesser parameter than basic Model(Net2). <br>
* Introduce a Batch Normalization after every convolution layer (Conv2d->BN-Relu) except last 2 conv2d layers <br>
* Apply Max Pooling and Adaptive Average Pooling <br>
## Results:
* Parameters: 17k (17,354) <br>
* Best Training Accuracy: 99.98 <br>
* Best Test Accuracy: 99.40 <br>
## Analysis:
* Net3 is much robust light-weight model with 17k parameters that consistently produces test 
  accuracy from ~ 99.25 to 99.4% range.<br>
* But the train model starts over fitting after it reached the max train accuracy of 99.98% and 
  no further improvements is possible.<br>
* Need to add drop out and see the model performance <br>
* 
##  Net4 MNIST classifier   ##
![Net4.png](Net4.png)
## Target:
* Create light-weight model (Net4) with lesser parameters(14.5k) than basic Model(Net3 with 17,
  354 parameters). <br>
* Introduce Drop out of 0.3 after every convolution transition block. <br>
* Reduce the max kernel parameter 20 against 24 in the previous model(Net3), this reduces the overall trainable parameters. <br>
## Results:
* Parameters: 14.5k (14,450) <br>
* Best Training Accuracy: 99.19 <br>
* Best Test Accuracy: 99.42 <br>
## Analysis:
* Net4 is much robust light-weight model with 14.5k parameters and consistently produced test 
  accuracy of ~ 99.35 to 99.43% in multiple epochs.<br>
* After added dropout, the over fitting problem was resolved, the max training accuracy is 99.
  19% and there is scope for additional improvement in training and to increase test accuracy further.<br>
* 
##  Net5 MNIST classifier   ##
![img.png](Net5.png)
## Target:
* Create a light-weight model (Net4) with lesser parameter(14.5k) than basic Model(Net4). <br>
* Introduce Drop out of 0.3 after every convolution transition block. <br>
## Results:
* Parameters: 12,122 <br>
* Best Training Accuracy: 98.76 <br>
* Best Test Accuracy: 99.43 <br>
## Analysis:
* Net5 is robust light-weight model with 12,122 parameters and consistently produced test 
  accuracy of ~ 99.35 to 99.43% in multiple epochs.<br>
* After added dropout, the over fitting problem was resolved, the max training accuracy is 99.
  43% and there is scope for additional improvement in training and to increase test accuracy further.<br>

##  Net6 MNIST classifier   ##
![Net6.png](Net6.png)
## Target:
* Create a light-weight model (Net6) with lesser parameters(<8k). <br>
* Reduce drop out value from 0.2 to 0.1 after every convolution layer. <br>
* Introduce Image Augmentation, such as RandomRotation (Identified from heuristics 5-7% 
  rotation would have provided additional scope for learning) and ColorJitter (Increase in 
  brightness and contrast). <br>

## Results:
* Parameters: 7,664 <br>
* Best Training Accuracy: 98.93 <br>
* Best Test Accuracy: 99.4 <br>
* Total Num of epochs: 14 <br>
## Analysis:
* Net6 is a most robust light-weight model created with 7.6k parameters and reached target test 
  accuracy 99.4% at 14th epoch with train accuracy of 98.93. <br>
* This model is not over fitting and there is additional scope for improvement and to boost the 
  test accuracy further. <br>
* Drop out after every convolution layer solved the over fitting problem. <br>
* Image augmentation helped to create under-fitting model with more option to improve the test 
  accuracy with additional training. <br>



