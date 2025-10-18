#this is for normalizing the pixel values

import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from PIL import Image
import numpy as np
import os



transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_set=datasets.ImageFolder("C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\split\\train",transform=transform)
test_set=datasets.ImageFolder("C:\\Users\\farza\\PycharmProjects\\Cervical-Cancer-Classification-Using-Deep-Learning\\data\\split\\test",transform=transform)


train_loader=DataLoader(train_set,batch_size=32,shuffle=True)
test_loader=DataLoader(test_set,batch_size=32,shuffle=True)
