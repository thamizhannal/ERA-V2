# Session8 Assignment CIFAR10 Classifier with 70% test accuracy in 50k parameters 

### Objective: ###
Given below is the objective of this assignment <br>
Your Assignment is: <br>
* Change the dataset to CIFAR10 
* Make this network: <br>
* C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11 <br>
* cN is 1x1 Layer <br>
* Keep the parameter count less than 50000 <br>
* Max Epochs is 20
* You are making 3 versions of the above code (in each case achieve above 70% accuracy):
* Network with Group Normalization
* Network with Layer Normalization
* Network with Batch Normalization
* Share these details
* Training accuracy for 3 models
* Test accuracy for 3 models
* Find 10 misclassified images for the BN, GN and LN model, and show them as a 5x2 image matrix in 3 separately annotated images.
* write an explanatory README file that explains:
* what is your code all about,
* your findings for normalization techniques,
* add all your graphs
* your collection-of-misclassified-images 

## Train Vs Validation Accuracy ##
| Moden Name                             | Train Accuracy | Test Accuracy |
|:---------------------------------------|:--------------:|--------------:|
| CIFAR10 Batch Normalization classifier |     73.39%     |        74.49% |
| CIFAR10 Layer Normalization classifier |     66.57%     |        70.43% |
| CIFAR10 Batch Normalization classifier |     67.84%     |        71.94% |

## Findings
* CIFAR10 Batch Normalization classifier has achieved the highest test accuracy among the all other 
normalization methods, But it consumes more time for training.

* CIFAR10 Layer Normalization classifier has accuracy between Batch Normalization and group 
normalization and take lesser training time.

* CIFAR10 Group Normalization classifier has slightly lower test accuracy than Batach Norm, but 
higher than layer norm and takes much lesser training time. 

## Missclassified Images

* **CIFAR10 Batch Normalization classifier**
  ![https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_BN_missclass.png](https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_BN_missclass.png)
 
* **CIFAR10 Layer Normalization classifier**
  ![https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_LN_missclass.png](https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_LN_missclass.png)
* **CIFAR10 Group Normalization classifier**
  ![https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_gn_missclass.png](https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_gn_missclass.png)
 

## Model Metrics
* **CIFAR10 Batch Normalization classifier**
  ![https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_BN_metrics.png](https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_BN_metrics.png)
  
* **CIFAR10 Layer Normalization classifier**
  ![https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_LN_metrics.png](https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_LN_metrics.png)
* **CIFAR10 Group Normalization classifier**
  ![https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_gn_metrics.png](https://raw.githubusercontent.com/thamizhannal/ERA-V2/main/s8/images/cifar10_gn_metrics.png)
* 
