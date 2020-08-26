import torch
import torch.nn as nn
import numpy as np
import math
import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import os
from datetime import datetime
import time
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import dataprocessing
from torch.nn import CTCLoss
from crnn import CRNN
# pasrsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epoch',type=int, default = 1000,help="epoch to start training")
parser.add_argument('--dataroot',type= str, default="mnt/ramdisk/max/90kDICT32px",help="directory to dataset")
parser.add_argument('--savedmodel',type=str,default="save",help="directory to saved model")
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--alphabet',type=str,default='0123456789abcdefghijklmnopqrstuvwxyz')
opt = parser.parse_args()

cuda = torch.cuda.is_available()
device= torch.device('cuda')

#intialize model
model = CRNN()
model.cuda()
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02).cuda()
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0).cuda()
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02).cuda()
        torch.nn.init.constant_(m.bias.data, 0.0).cuda()
        # weitghts initalize
weights_init(model)


#load saved model
if opt.savedmodel!="save":
    model.load_state_dict(torch.load("savedmodel/%s.pth"%opt.savedmodel))
#create optimizer 
optimizer = torch.optim.Adadelta(model.parameters()) 
#dataloader 
traindata = DataLoader(dataprocessing.ImageDataset(opt.dataroot,mode="train"),batch_size=opt.batchsize,shuffle= True)
print(len(traindata))
testdata = DataLoader(dataprocessing.ImageDataset(opt.dataroot,mode="test"),batch_size=opt.batchsize, shuffle=True)
print(len(testdata))
# Loss function
lossfunction = CTCLoss()
process = dataprocessing.ProcessText(opt.alphabet)
# tensorboard
writer = SummaryWriter()
def test():
    print("------TEST-------")
    teststep=1
    correct = 0
    number = 0
    for i, batch in enumerate(testdata):
        raw_text =batch["label"]
        encode_text,length = process.encodetext(raw_text)
        encode_text = Variable(encode_text).to(device)
        image = Variable(batch["image"]).to(device)
        model.eval()
        output = model(image)
        output_size=Variable(torch.IntTensor([output.size(0)]*opt.batchsize)).to(device)        
        loss = lossfunction(output,encode_text,output_size,length)
        _,output=output.max(2)
        output = output.transpose(1,0)
        outputtext=[]
        for i in range(0, output.size(0)):
            decode_text = process.decodetext(output[i])
            outputtext+=[decode_text]
            number+=1
            if decode_text == raw_text[i]:
                correct+=1
        accuracy = float(correct/number)
        teststep+=1
        print("-----Test-----step:%d/%d----loss value:%f-----accuracy:%f"%(teststep,len(testdata),loss,accuracy))
        writer.add_scalar("Testing Loss",loss,step)
        writer.add_scalar("Testing accuracy",accuracy,epoch)
step=1
for epoch in range(0,opt.epoch):    
    correct = 0
    number = 0
    for i,batch in enumerate(traindata):
        raw_text = batch["label"]
        encode_text,length = process.encodetext(raw_text) 
        encode_text = Variable(encode_text).to(device)      
        image = Variable(batch["image"]).to(device)
        model.train()
        output = model(image)
        optimizer.zero_grad()
        output_size = Variable(torch.IntTensor([output.size(0)]*opt.batchsize)).to(device)
        loss = lossfunction(output,encode_text,output_size,length)
        loss.backward()
        optimizer.step()
        _,output = output.max(2)
        output = output.transpose(1,0)
        outputtext=[]
        for i in range(0,output.size(0)):
            decode_text = process.decodetext(output[i])
            outputtext+=[decode_text]
            number+=1
            if decode_text == raw_text[i]:
                correct+=1
        accuracy = float(correct/number)        
        print('epoch:%d-----step:%d/%d-----loss value:%f-----accuracy:%f\n'%(epoch,step,len(traindata),loss,accuracy))
        print(raw_text,'\n')
        print(outputtext,'\n')
        writer.add_scalar("Training Loss",loss,step)        
        step+=1
        if step%100==0:
            test()
            torch.save(model.state_dict(),"savedmodel/epoch%d-step%d.pth"%(epoch,step))
        writer.add_scalar("Training Accuracy",accuracy,epoch)