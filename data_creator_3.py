import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


input_size=784 # 28*28 pixels of image flattened to 1D array of 784
classes=10 # 10 classes of digits
batch_size=100

# DATA LOADING 
training_dataset = datasets.MNIST(root="data",train=True,download=True,transform=ToTensor(),)
test_dataset = datasets.MNIST(root="data",train=False,download=True,transform=ToTensor(),)
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# DATA CREATION AND SAVING 

import random
import numpy as np
from PIL import Image

for c in range(10000):
    # choose i, j randomly from 0 to 60000
    i=random.randint(0,60000)
    j=random.randint(0,60000)
    k=random.randint(0,60000)
    image1=training_dataset[i][0][0]
    image2=training_dataset[j][0][0]
    image3=training_dataset[k][0][0]
    # combine the two images by concatenating
    combined_image=torch.cat((image1,image2),1) # 1 is for horizontal concatenation, 0 is for vertical concatenation
    combined_image=torch.cat((combined_image,image3),1)
    combined_image_np = combined_image.numpy()
    combined_image_pil = Image.fromarray((combined_image_np * 255).astype(np.uint8))
    combined_image_pil.save(f"./data/mnist_3/image_3_{c+1}.jpg")
