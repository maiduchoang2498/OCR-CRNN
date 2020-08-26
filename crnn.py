import torchvision
import cv2 as cv
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import dataprocessing
transform_ =[
   transforms.Resize((32,100),Image.BICUBIC), #resize the image to 100*32
   transforms.ToTensor(),# transform image to tensor for training Generators and discriminators
]
class BidirectionalLSTM(nn.Module):
    def __init__(self,in_channel,hidden_units,out_channel):
        super(BidirectionalLSTM,self).__init__()
        self.LSTM=nn.LSTM(in_channel,hidden_units,bidirectional=True)
        self.embedding = nn.Linear(hidden_units*2,out_channel)
    def forward(self,input):
        x,_= self.LSTM(input)
        T,b,h = x.size() # x.size()= (seq_len, batch, num_directions * hidden_size)
        x=x.view(T*b,h)
        x= self.embedding(x)
        x = x.view(T,b,-1)
        return x
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN,self).__init__()
        self.extractfeatures = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2),          
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512,512,kernel_size=2,stride=1,padding=0),
            nn.LeakyReLU(0.2),
            
        )
        self.RNN = nn.Sequential(
            BidirectionalLSTM(512,256,256),
            BidirectionalLSTM(256,256,37) 
            # 37 la so class can phan loai gom 10 chu so, 26 ki tu bang chu cai va 1 ki tu trong
    )
    def forward(self,input):
        x =self.extractfeatures(input)
        b,c,h,w= x.size() # take size as batch,channel,height,width
        x = x.squeeze(2) # squeeze x to (b,c,h*w)
        x=x.permute(2,0,1) # resize x to (seq_len, batch, input_size) for feeding to RNN
        
        output = self.RNN(x)
        return output 

# D = CRNN()
# D.cuda()
# x = torch.Tensor(3,1,32,100).cuda()
# x = D(x)
# print(x.size())

# c= nn.LSTM(10,20,2) #input size, output size, number layers
# input =torch.rand(5,3,10) # (seq_len, batch, input_size)
# h0 = torch.rand(2,3,20) #(num_layers * num_directions, batch, hidden_size)
# c0 = torch.rand(2,3,20)
# output, (hn,cn) = c(input,(h0,c0))