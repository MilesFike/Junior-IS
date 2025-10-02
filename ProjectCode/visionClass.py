import torch.nn as nn
import torch
#The inputs for this model are based on 
#Data process for training and testing
class Net(nn.Module): #nn.Module is the base for Pytorch models. I am just filling in the gaps to define convolution layers.
    def __init__(self):
        super(Net, self).__init__() 
        self.convolute = nn.Conv2d(1, 10, stride=1, kernel_size=5) #1 input channel because it is gray scale. 10 output because 10 is my lucky number. Stride is step size. Kernel is 3rd num (28-5) + 1 = 24, 10 * 24^2
        self.pool = nn.MaxPool2d(stride=2, kernel_size=2) #10 * 12^2
        self.convolute2 = nn.Conv2d(10, 20, stride=1, kernel_size = 5) #Convolution layers (12-5) + 1 = 8
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2) #Pooling layers 20 * 4 ^2
        self.fullcon1 = nn.Linear(20 *4 *4, 104) #Fully connected layers
        self.fullcon2 = nn.Linear(104, 52)
        self.fullcon3 = nn.Linear(52, 26)

    def forward(self, x):
        x = self.pool(torch.relu(self.convolute(x)))   # Common activation for nn, turns negatives to 0
        x = self.pool(torch.relu(self.convolute2(x)))   
        x = x.view(-1, 20 *4*4) # prep for fullcon1
        x = torch.tanh(self.fullcon1(x)) #Tahn limits to between one and negative 1
        x = torch.tanh(self.fullcon2(x))
        x = self.fullcon3(x)
        return x
net = Net()