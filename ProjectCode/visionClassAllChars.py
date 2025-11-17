import torch.nn as nn #nn is for neural networks
import torch
#The inputs for this model are based on 
#Data process for training and testing
class Net(nn.Module): #nn.Module is the base for Pytorch models. I am just filling in the gaps to define convolution layers.
    def __init__(self):
        super(Net, self).__init__() 
        
        #Model 1
        self.convolute = nn.Conv2d(1, 32, stride=1, kernel_size=5) #1 input channel because it is gray scale. 10 foroutput. Stride is step size. Kernel is 3rd num (28-5) + 1 = 24, 10 * 24^2
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2) #10 * 12^2
        self.convolute2 = nn.Conv2d(32, 64, stride=1, kernel_size = 5) #Convolution layers (12-5) + 1 = 8
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2) #Pooling layers 40 * 4 ^2
        self.fullcon1 = nn.Linear(64 *4 *4, 256) #Fully connected layers
        self.fullcon2 = nn.Linear(256, 104)
        self.fullcon3 = nn.Linear(104, 62)

    def forward(self, x):
        x = self.pool(torch.relu(self.convolute(x)))   # Common activation for nn, turns negatives to 0
        x = self.pool(torch.relu(self.convolute2(x)))   
        #Model 1
        x = x.view(-1, 64 *4*4) # prep for fullcon1
        #Model 2
        #x = x.view(-1, 64 *4*4) # prep for fullcon1
        x = torch.tanh(self.fullcon1(x)) #Tahn limits to between one and negative 1
        x = torch.tanh(self.fullcon2(x))
        x = self.fullcon3(x)
        return x
net = Net()