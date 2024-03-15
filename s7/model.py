# import packages to make neural network model
import torch.nn as nn
import torch.nn.functional as F

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 512, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(512, 1024, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 |
        self.conv7 = nn.Conv2d(1024, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) #input=28x28x1 output=28x28x8 -? OUtput? RF =3x3
        #self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # input=28x28x8 output=28x28x16 | RF=5x5
        #self.bn2 = nn.BatchNorm2d(16) 
        self.pool1 = nn.MaxPool2d(2, 2) # input=28x28x16 output=14x14x16 | RF = 10x10 
        #self.dp1 =nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1) # input=14x14x16 output=14x14x24 | RF=12x12
        #self.bn3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 28, 3, padding=1) ## input=14x14x24 output=14x14x28 | RD=14x14
        #self.bn4 = nn.BatchNorm2d(28)
        self.pool2 = nn.MaxPool2d(2, 2) # input=14x14x28 output=7x7x28 | RF=28x28
        #self.dp2 =nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(28, 24, 3) # input=7x7x28 output=7x7x24 | RF=30x30
        #self.bn5 = nn.BatchNorm2d(24)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv6 = nn.Conv2d(24, 16, 3) # input=7x7x24 output=5x5x16 | RF=32x32
        self.conv7 = nn.Conv2d(16, 10, 3) # input=5x5x16 output=3x3x10 | RF=34x34

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)        
        return F.log_softmax(x)


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) #input=28x28x1 output=28x28x8 -? OUtput? RF =3x3
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # input=28x28x8 output=28x28x16 | RF=5x5
        self.bn2 = nn.BatchNorm2d(16) 
        self.pool1 = nn.MaxPool2d(2, 2) # input=28x28x16 output=14x14x16 | RF = 10x10 
        #self.dp1 =nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(16, 20, 3, padding=1) # input=14x14x16 output=14x14x20 | RF=12x12
        self.bn3 = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(20, 24, 3, padding=1) ## input=14x14x20 output=14x14x24 | RD=14x14
        self.bn4 = nn.BatchNorm2d(24)
        self.pool2 = nn.MaxPool2d(2, 2) # input=14x14x24 output=7x7x24 | RF=28x28
        #self.dp2 =nn.Dropout(0.2)
        self.conv5 = nn.Conv2d(24, 20, 3) # input=7x7x24 output=7x7x20 | RF=30x30
        self.bn5 = nn.BatchNorm2d(20)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv6 = nn.Conv2d(20, 16, 3) # input=7x7x20 output=5x5x16 | RF=32x32
        self.conv7 = nn.Conv2d(16, 10, 3) # input=5x5x16 output=3x3x10 | RF=34x34

    def forward(self, x):
        x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = F.relu(self.conv6(F.relu(self.bn5(self.conv5(x)))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) #input=28x28x1 output=28x28x8 -? OUtput? RF =3x3
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # input=28x28x8 output=28x28x16 | RF=5x5
        self.bn2 = nn.BatchNorm2d(16) 
        self.pool1 = nn.MaxPool2d(2, 2) # input=28x28x16 output=14x14x16 | RF = 10x10 
        self.dp1 =nn.Dropout(0.3)
        
        self.conv3 = nn.Conv2d(16, 20, 3, padding=1) # input=14x14x16 output=14x14x20 | RF=12x12
        self.bn3 = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(20, 16, 3, padding=1) ## input=14x14x20 output=14x14x24 | RD=14x14
        self.bn4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2) # input=14x14x24 output=7x7x24 | RF=28x28
        self.dp2 =nn.Dropout(0.3)
        
        self.conv5 = nn.Conv2d(16, 20, 3) # input=7x7x24 output=7x7x20 | RF=30x30
        self.bn5 = nn.BatchNorm2d(20)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.conv6 = nn.Conv2d(20, 16, 3) # input=7x7x20 output=5x5x16 | RF=32x32
        self.conv7 = nn.Conv2d(16, 10, 3) # input=5x5x16 output=3x3x10 | RF=34x34

    def forward(self, x):
        x = self.dp1(self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))))
        x = self.dp2(self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x))))))))
        x = F.relu(self.conv6(F.relu(self.bn5(self.conv5(x)))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) #input=28x28x1 output=28x28x8 -? OUtput? RF =3x3
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # input=28x28x8 output=28x28x16 | RF=5x5
        self.bn2 = nn.BatchNorm2d(16) 
        self.pool1 = nn.MaxPool2d(2, 2) # input=28x28x16 output=14x14x16 | RF = 10x10 
        self.dp1 =nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(16, 12, 3, padding=1) # input=14x14x16 output=14x14x12 | RF=12x12
        self.bn3 = nn.BatchNorm2d(12)
        self.conv4 = nn.Conv2d(12, 16, 3, padding=1) ## input=14x14x12 output=14x14x16 | RD=14x14
        self.bn4 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2) # input=14x14x24 output=7x7x24 | RF=28x28
        self.dp2 =nn.Dropout(0.2)        
        self.conv5 = nn.Conv2d(16, 20, 3) # input=7x7x16 output=7x7x20 | RF=30x30
        self.bn5 = nn.BatchNorm2d(20)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.conv6 = nn.Conv2d(20, 16, 3) # input=7x7x20 output=5x5x16 | RF=32x32
        self.conv7 = nn.Conv2d(16, 10, 3) # input=5x5x16 output=3x3x10 | RF=34x34

    def forward(self, x):
        x = self.pool1(self.dp1(F.relu(self.bn2(self.conv2(self.dp1(F.relu(self.bn1(self.conv1(x)))))))))
        x = self.pool2(self.dp2(F.relu(self.bn4(self.conv4(self.dp2(F.relu(self.bn3(self.conv3(x)))))))))
        x = F.relu(self.conv6(self.dp2(F.relu(self.bn5(self.conv5(x))))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False) #input=28x28x1 output=28x28x8 -? OUtput? RF =3x3
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1, bias=False) # input=28x28x8 output=28x28x12 | RF=5x5
        self.bn2 = nn.BatchNorm2d(12) 
        self.pool1 = nn.MaxPool2d(2, 2) # input=28x28x12 output=14x14x12 | RF = 10x10 
        self.dp1 =nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1, bias=False) # input=14x14x12 output=14x14x16 | RF=12x12
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 10, 3, padding=1, bias=False) ## input=14x14x16 output=14x14x10 | RD=14x14
        self.bn4 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(2, 2) # input=14x14x10 output=7x7x10 | RF=28x28
        self.dp2 =nn.Dropout(0.1)        
        self.conv5 = nn.Conv2d(10, 14, 3, bias=False) # input=7x7x10 output=7x7x16 | RF=30x30
        self.bn5 = nn.BatchNorm2d(14)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.conv6 = nn.Conv2d(14, 10, 3) # input=7x7x16 output=5x5x10 | RF=32x32
        self.conv7 = nn.Conv2d(10, 10, 3) # input=5x5x10 output=3x3x10 | RF=34x34

    def forward(self, x):
        x = self.pool1(self.dp1(F.relu(self.bn2(self.conv2(self.dp1(F.relu(self.bn1(self.conv1(x)))))))))
        x = self.pool2(self.dp2(F.relu(self.bn4(self.conv4(self.dp2(F.relu(self.bn3(self.conv3(x)))))))))
        x = self.conv6(F.relu(self.bn5(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False) #input=28x28x1 output=28x28x8 -? OUtput? RF =3x3
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 12, 3, padding=1, bias=False) # input=28x28x8 output=28x28x12 | RF=5x5
        self.bn2 = nn.BatchNorm2d(12) 
        self.pool1 = nn.MaxPool2d(2, 2) # input=28x28x12 output=14x14x12 | RF = 10x10 
        self.dp1 =nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2d(12, 16, 3, padding=1, bias=False) # input=14x14x12 output=14x14x16 | RF=12x12
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 10, 3, padding=1, bias=False) ## input=14x14x16 output=14x14x10 | RD=14x14
        self.bn4 = nn.BatchNorm2d(10)
        self.pool2 = nn.MaxPool2d(2, 2) # input=14x14x10 output=7x7x10 | RF=28x28
        self.dp2 =nn.Dropout(0.1)        
        self.conv5 = nn.Conv2d(10, 14, 3, bias=False) # input=7x7x10 output=7x7x16 | RF=30x30
        self.bn5 = nn.BatchNorm2d(14)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.conv6 = nn.Conv2d(14, 10, 3) # input=7x7x16 output=5x5x10 | RF=32x32
        self.conv7 = nn.Conv2d(10, 10, 3) # input=5x5x10 output=3x3x10 | RF=32x32

    def forward(self, x):
        x = self.pool1(self.dp1(F.relu(self.bn2(self.conv2(self.dp1(F.relu(self.bn1(self.conv1(x)))))))))
        x = self.pool2(self.dp2(F.relu(self.bn4(self.conv4(self.dp2(F.relu(self.bn3(self.conv3(x)))))))))
        x = self.conv6(F.relu(self.bn5(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)