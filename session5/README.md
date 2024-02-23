<H1> MNIST Handwritten Digit Recognition using CNN(Convolution Neural Network) in PyTorch </H1> 
This repository consists of an MNIST handwritten digit classifier implemented using Convolution Neural Network in PyTorch.
<br>
<H2> Overview: </H2>

<H3> MNIST Data Set </H3> 
MNIST contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images are grayscale, 28x28 pixels, and centred on reducing preprocessing and getting started quicker. 
<BR>

<H3> List Of Files </H3>
<h3>model.py</h3>  <br>
This defines neural network architecture that consists of two blocks of convolution layers 
followed by fully connected layers. <br>

**Block1:** <br>
This consists of two conv2d layers convolving on 3x3 kernel followed by
Max pooling of 2x2 with stride 2. 

conv2d-1: input: 28x28x1  output: 26x26x32 output RF:3x3  <br>
conv2d-2: input: 26x26x32 output:24x24x64 output RF:5x5 <br>
MP-1: Max Pool with stride=2 output: 12x12x64 output RF: 10x10 <br>

**Block2:** <br>
This consists of two conv2d layers convolving on 3x3 kernel followed by
Max pooling of 2x2 with stride 2.

conv2d-3: Input: 12x12x64 output:10x10x128 output RF:12x12 <br>
conv2d-4: Input: 10x10x128 output: 8x8x256 output RF:14x14** <br>
MP-2: Max Pooling with stride=2 - Input: 8x8x256 output: 4x4x256 output RF:28x28 <br>

Fully Connected Layer: <BR>
Two FC layers were used, The first one consisted of 4096 neurons that 
downsized into 50 neural, and later one reduced that into 10 predictions.
```python
   self.fc1 = nn.Linear(256*4*4, 50, bias=False)
   self.fc2 = nn.Linear(50, 10, bias=False)
```
<h3>utils.py</H3> <BR>
* apply_transformations: a method that defines the required transformation for training and test datasets. <BR>
* create_train_test_dataset : this method downloads the train and test data set and applies corresponding transformations. <BR>
* train: A method that performs feed-forward network training and back-propagation
    and records metrics such as train accuracy and loss <BR>
* test: A method that performs feed-forward network testing
    and records metrics such as test accuracy and loss <BR>
* get_correct_pred_count :A method returns prediction count as label <BR>

<br>
<H3> S5.ipynb </H3> <BR>
This is the main file for the MNIST digit recognition classifier implemented using CNN and fully connected layers. <BR>

* Defines necessary pytorch packages
* Loads transformation functions
* Creates train and test data set
* Plot a sample digits
* Run the Network with records train and test accuracies and losses
* Plot the train and test accuracies and losses against epochs

<u><h3> Run Command: </H3> </u><br>
Run the following command in Linux/Mac terminal <br>
**Install Prerequisites: pip install -r requirements.txt <br>
To Run ipynb: jupyter nbconvert --execute S5.ipynb <br>
To Run ipynb as python file: jupyter nbconvert --to python S5.ipynb <br>**
 
