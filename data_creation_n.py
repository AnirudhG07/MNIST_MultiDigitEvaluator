import torch 
import torchvision
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
import torch
import csv, os
import shutil

no_of_images = 10000

mnist_directorytory_path = f"./data/mnist_{n}"

# Remove the directory if it exists
if os.path.exists(mnist_directorytory_path):
    shutil.rmtree(mnist_directorytory_path)

# Create the directory
os.makedirs(mnist_directorytory_path)

# Function to create CSV file
def create_csv_file(labels, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Id', 'Label'])

        for i in range(len(labels)):
            csvwriter.writerow([i + 1, labels[i]])

labels_list = []

for c in range(no_of_images):
    digit = []
    for i in range(n):
        digit.append(random.randint(0, 59999))

    images = []
    labels = [] 
    for i in digit:
        image, label = training_dataset[i]
        images.append(training_dataset[i][0][0])
        labels.append(label)

    combined_image=images[0]
    for i in range(1,n):
        combined_image=torch.cat((combined_image,images[i]),1)
    combined_image_np = combined_image.numpy()
    combined_image_pil = Image.fromarray((combined_image_np * 255).astype(np.uint8))
    combined_image_pil.save(f"./data/mnist_{n}/image_{n}_{c+1}.jpg")

    flat_labels = ''.join(map(str, labels))
    labels_list.append(flat_labels)

# Create CSV file
csv_filename = f'./data/mnist_{n}_labels.csv'
create_csv_file(labels_list, csv_filename)

print(f"CSV file '{csv_filename}' created successfully.")
