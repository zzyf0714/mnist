'''MINISTIN.py'''
'''
• 数据：MNIST data set

• 本题目考察如何设计并实现一个简单的图像分类器。设置本题目的目的如下：
– 理解基本的图像识别流程及数据驱动的方法（训练、预测等阶段）
– 理解训练集/验证集/测试集的数据划分，以及如何使用验证数据调整模型的超参数
– 实现一个Softmax分类器
– 实现一个全连接神经网络分类器
– 理解不同的分类器之间的区别，以及使用不同的更新方法优化神经网络

• 附加题： 
– 尝试使用不同的损失函数和正则化方法，观察并分析其对实验结果的影响 (+5 points)
– 尝试使用不同的优化算法，观察并分析其对训练过程和实验结果的影响 (如batch GD, online GD, mini-batch GD, SGD, 或其它的优化算法，如Momentum, Adsgrad, Adam, Admax) (+5 points)
• 补充：MINST是一个手写数字数据集，包括了若干手写数字体及其对应的数字，共60000个训练样本，10000个测试样本。每个手写数字被表示为一个28*28的向量。  
'''
# ------------------数据预处理------------------------------
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
import struct

train_data=torchvision.datasets.MNIST(
    root='D:\作业\深度学习与神经网络\作业一',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_data=torchvision.datasets.MNIST(
    root='D:\作业\深度学习与神经网络\作业一',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# train_load=DataLoader(dataset=train_data,batch_size=1000,shuffle=True)
# test_load=DataLoader(dataset=test_data,batch_size=500,shuffle=True)
# train_x,train_y=next(iter(train_load))

# 读取原始数据并进行预处理
def data_fetch_preprocessing():
    train_image = open('train-images-idx3-ubyte', 'rb')
    test_image = open('t10k-images-idx3-ubyte', 'rb')
    train_label = open('train-labels-idx1-ubyte', 'rb')
    test_label = open('t10k-labels-idx1-ubyte', 'rb')

    magic, n = struct.unpack('>II',train_label.read(8))
    # 原始数据的标签
    y_train_label = np.array(np.fromfile(train_label,dtype=np.uint8), ndmin=1)
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 0.99

    # 测试数据的标签
    magic_t, n_t = struct.unpack('>II',test_label.read(8))
    y_test = np.fromfile(test_label,dtype=np.uint8).reshape(10000, 1)
    # print(y_train[0])
    # print(len(labels))
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784).T

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test), 784).T
    # print(x_train.shape)
    # 可以通过这个函数观察图像
    # data=x_train[:,0].reshape(28,28)
    # plt.imshow(data,cmap='Greys',interpolation=None)
    # plt.show()
    # x_train = x_train / 255 * 0.99 + 0.01
    # x_test = x_test / 255 * 0.99 + 0.01

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test

#data_fetch_preprocessing()


# ------------------梯度------------------------------






# -----------------激活函数-------------------------

def softmax(X):
    '''
    param：X为一个向量
    return：返回softmax激活值
    '''
    X_exp=torch.exp(X)
    exp_sum=X_exp.sum(dim=1,keepdim=True) 
    return X_exp/exp_sum

# -----------------前向传播----------------------------




# -----------------反向传播----------------------------




# -----------------交叉熵损失函数-----------------------




# -----------------训练模型----------------------------





# -----------------预测----------------------------