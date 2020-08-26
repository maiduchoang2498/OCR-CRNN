import glob
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image
import random
import os 
import math
import cv2 as cv
import torch
import collections
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, root,mode="train",unalinged=False):
        self.transform =transforms.Compose([
                transforms.Resize((32,100),Image.BICUBIC), #resize the image to 100*32
                transforms.ToTensor()# transform image to tensor 
        ])
        self.directories = sorted(open("%s/annotation_%s.txt"%(root,mode),'r'))
        self.unalinged=unalinged
        self.root=root
        #return a list contains file name    
    def __getitem__(self,index):
        link = str(self.directories[random.randint(0,len(self.directories)-1)][1:])
        link = link.split(" ")[0]
        link=self.root+link
        image = Image.open(link)
        #open Image
        tensor = self.transform(image)
        #transform image to tensor and fit size
        label = link.split("_")[1].lower() #label is name of image
        # take the label
        return {"image":tensor,"label":label}
    def __len__(self):
        return len(self.directories)
class ProcessText(object):
    def __init__(self,alphabet):
        self.alphabet=alphabet
    def encodetext(self,text):#do pytorch tensor khong ho tro kieu string nen phai convert text ra chu so  
        # text = [text[0],text[1],...,text[batch_size-1]]
        encode_text = []
        length = []
        for s in text:
            length+= [len(s)]
            x=[]
            for char in s:
                x+=[self.alphabet.find(char)+1] #tra ve chi so cua ki tu char trong str alphabet dc nhap vao 
            encode_text+=x
        return (torch.IntTensor(encode_text),torch.IntTensor(length))
    def decodetext(self,text):#ket qua cua model la so thu tu cac nhan dc sinh ra tu 1-nclass, phai chuyen ve text de hien thi
        decodetext=[]
        x=text.detach()    
        y=""
        if x[0]!=0:
            y+=self.alphabet[x[0]-1]
        for j in range(1,len(text)):
            if x[j]!=0 and x[j]!=x[j-1]:
                y+=self.alphabet[x[j]-1]
        return y


    

