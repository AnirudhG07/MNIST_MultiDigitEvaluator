import torch 
=import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


input_size=784 # 28*28 pixels of image flattened to 1D array of 784
classes=10 # 10 classes of digits
batch_size=100
# DATA LOADING 

training_dataset = datasets.MNIST(root="data",train=True,download=True,transform=ToTensor(),)
test_dataset = datasets.MNIST(root="data",train=False,download=True,transform=ToTensor(),)
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# VAUE OF N, either input or set yourself beforehand
n=int(input("Enter the number of digits inside image: "))
print(n)

# DATA CREATION 

import random
from PIL import Image
import numpy as np

no_of_images=1 # Change according to your wish
for c in range(no_of_images):
    digit=[]
    for i in range(n):
        digit.append(random.randint(0,60000))
    images=[]
    for i in digit:
        images.append(training_dataset[i][0][0])
    combined_image=images[0]
    for i in range(1,n):
        combined_image=torch.cat((combined_image,images[i]),1)
    combined_image_np = combined_image.numpy()
    combined_image_pil = Image.fromarray((combined_image_np * 255).astype(np.uint8))
    combined_image_pil.save("image.jpg")
