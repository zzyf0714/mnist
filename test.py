import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

# def forward_hiddenlayer(input_data,weight,b):
#         z=np.add(np.dot(weight,input_data),b)
#         return z,softmax(z)
print(softmax([1,2]))
