# import packages to make neural network model
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        # input: 28x28x1  output: 26x26x32 output RF:3x3 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        
        # input: 26x26x32 output:24x24x64 output RF:5x5
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Max Pool with stride=2 output: 12x12x64 output RF: 10x10
        
        # Input: 12x12x64 output:10x10x128 output RF:12x12
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        
        # Input: # 10x10x128 output: 8x8x256 output RF:14x14
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        
        # Max Pooling with stride=2 - Input: 8x8x256 output: 4x4x256 output RF:28x28
        
        self.fc1 = nn.Linear(256*4*4, 50, bias=False)
        self.fc2 = nn.Linear(50, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 256*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)