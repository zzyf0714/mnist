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

def sigmoid(x):
        return 1 / (1 + np.exp(-x))
b=np.array([0.01,0.01,0.01,0.01,0,99,0.01,0.01,0.01,0.01,0.01])
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])


random_permutation = np.random.permutation(data.shape[0])[:3]
shuffled_data = data[ np.random.permutation(data.shape[0])[:3],:]



import pandas as pd

# 创建一个DataFrame对象
data = {'姓名': ['张三', '李四', '王五'], '年龄': [20, 25, 30], '性别': ['男', '女', '男']}
df = pd.DataFrame(data)

# 显示表格
print(df)
j=1
precision=95
print("%d precision:%d" % j % precision + "%")