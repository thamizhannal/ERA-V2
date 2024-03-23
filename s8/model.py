'''
Change the dataset to CIFAR10
Make this network:
C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11
cN is 1x1 Layer
Keep the parameter count less than 50000
Max Epochs is 20
You are making 3 versions of the above code (in each case achieve above 70% accuracy):
Network with Group Normalization
Network with Layer Normalization
Network with Batch Normalization
Share these details
Training accuracy for 3 models
Test accuracy for 3 models
Find 10 misclassified images for the BN, GN and LN model, and show them as a 5x2 image matrix in 3 separately annotated images. 
write an explanatory README file that explains:
what is your code all about,
your findings for normalization techniques,
add all your graphs
your collection-of-misclassified-images 
Upload your complete assignment on GitHub and share the link on LMS
'''
# import packages to make neural network model
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalizationMethod(self, normType, out_channels):
    if normType =='BN':
        return nn.BatchNorm2d(out_channels)
    elif normType == 'GN':
        return nn.GroupNorm(num_groups=4, num_channels=out_channels)
    elif normType == 'LN':
        return nn.GroupNorm(num_groups=1, num_channels=out_channels)

class CIFAR10Net(nn.Module):
        
    # C1 C2 c3 P1 C4 C5 C6 c7 P2 C8 C9 C10 GAP c11
    def __init__(self, normType='BN'):
        super(CIFAR10Net, self).__init__()
        # Input block input_size=32
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), padding=1, bias=False),
            normalizationMethod(self, normType=normType,out_channels=16),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) # output_size = 32 | RF = 3x3
        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            normalizationMethod(self,normType=normType,out_channels=24),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) # output_size = 32 | RF = 5x5
        

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=16, kernel_size=(1, 1), padding=1, bias=False),
            normalizationMethod(self,normType=normType,out_channels=16),
            nn.ReLU()
        ) # output_size = 32 | RF = 5x5        
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16

        # CONVOLUTION BLOCK 2

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            normalizationMethod(self,normType=normType,out_channels=24),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) # output_size = 16        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            normalizationMethod(self,normType=normType,out_channels=32),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) # output_size = 16 
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=40, kernel_size=(3, 3), padding=1, bias=False),
            normalizationMethod(self,normType=normType,out_channels=40),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) # output_size = 16 
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=24, kernel_size=(1, 1), padding=1, bias=False),
            normalizationMethod(self,normType=normType,out_channels=24),
            nn.ReLU()
        ) # output_size = 16 
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            normalizationMethod(self,normType=normType,out_channels=32),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) # output_size = 8        
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=28, kernel_size=(3, 3), padding=0, bias=False),
            normalizationMethod(self,normType=normType,out_channels=28),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) # output_size = 6
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=28, out_channels=24, kernel_size=(3, 3), padding=0, bias=False),
            normalizationMethod(self,normType=normType,out_channels=24),
            nn.ReLU()
        ) # output_size = 4
        
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2) #
        ) # output_size = 2
        
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=10, kernel_size=(2, 2), padding=0, bias=False),
        ) # output_size = 10 
        

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        
        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.convblock7(x)  
        x = self.pool2(x)

        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        
        x = self.gap(x)

        x = self.convblock11(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)