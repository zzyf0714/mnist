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


# 读取原始数据并进行预处理
def data_fetch_preprocessing():
    train_image = open('train-images-idx3-ubyte', 'rb')
    test_image = open('t10k-images-idx3-ubyte', 'rb')
    train_label = open('train-labels-idx1-ubyte', 'rb')
    test_label = open('t10k-labels-idx1-ubyte', 'rb')

    magic, n = struct.unpack('>II',train_label.read(8))
    # 原始数据的标签
    y_train_label = np.array(np.fromfile(train_label,dtype=np.uint8), ndmin=1)
    y_train = np.zeros((10, 60000)) 
    for i in range(60000):
         y_train[y_train_label[i]][i] = 1
    # y_train = np.ones((10, 60000)) * 0.01
    # for i in range(60000):
    #     y_train[y_train_label[i]][i] = 0.99

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
    x_train = x_train / 255 * 0.99 + 0.01
    x_test = x_test / 255 * 0.99 + 0.01

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test

#data_fetch_preprocessing()

#------------------搭建神经网络----------------------
'''
包括输入层，一层隐藏层，输出层
输入层节点：784 隐藏层节点：200 输出层节点：100
隐藏层使用 sigmoid激活函数
输出层使用 softmax激活函数
'''
class Nerual_Network(object):
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        '''
        :param inputnodes: 输入层结点数
        :param hiddennodes: 隐藏层结点数
        :param outputnodes: 输出层结点数
        :param learningrate: 学习率
        '''
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
        self.lambd=0.01
        # 输入层与隐藏层权重矩阵初始化
        self.w1 = np.random.randn(self.hiddennodes, self.inputnodes) * 0.01
        # 隐藏层与输出层权重矩阵初始化
        self.w2 = np.random.randn(self.outputnodes, self.hiddennodes) * 0.01
        # 构建第一层常量矩阵100 by 1 matrix
        self.b1 = np.zeros((200, 1))
        # 构建第二层常量矩阵 10 by 1 matrix
        self.b2 = np.zeros((10, 1))
        # 定义迭代次数
        self.epoch = 50
    # -----------------激活函数-------------------------

    def softmax(self,x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0)
        
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    # def sigmoid_grad(x):
    #     return x*(1-x)
    def sigmoid_grad(self,x):
        return x*(1-x)
    # -----------------前向传播----------------------------
    def forward_hiddenlayer(self,input_data,weight,b):
        z=np.add(np.dot(weight,input_data),b)
        return z,self.sigmoid(z)

    def forward_outlayer(self,input_data,weight,b):
        z=np.add(np.dot(weight,input_data),b)
        return z,self.softmax(z)
    
    # -----------------交叉熵损失函数-----------------------
    def crossEntropy(self,x,label_data,batch_size):
        #loss =-np.sum(label_data*np.log(softmax(x)))
        loss =-np.sum(label_data*np.log(x))/batch_size 
        return loss

    def crossEntropy_L2(self,x,label_data,batch_size,w1,w2):
        #loss =-np.sum(label_data*np.log(softmax(x)))
        loss =-np.sum(label_data*np.log(x))/batch_size 
        L2_regulation=self.lambd*(np.sum(np.square(w1)) + np.sum(np.square(w2)))/batch_size/2
        return loss

    def back_outlayer(self,out,label_data,forward_a,batch_size,w2):
        dz = out - label_data
        self.w2 -= self.learningrate * np.dot(dz, forward_a.T)/batch_size+ self.lambd / batch_size * w2
        self.b2 -= self.learningrate * (np.mean(dz,axis=1).reshape(-1,1))
        return dz
    def back_hiddenlay(self,grad_z,current_a,input_data,batch_size,w1):
        dz = np.dot(self.w2.T, grad_z) * current_a * (1.0 - current_a) #sigmoid的梯度
        self.w1 -= self.learningrate * np.dot(dz, (input_data).T)/batch_size + self.lambd / batch_size * w1
        self.b1 -= self.learningrate * (np.mean(dz,axis=1).reshape(-1,1))
        return dz
    # -----------------训练模型----------------------------
    def train(self, input_data, label_data,train_size,batch_size):
        batches_x = [input_data[:,i: i+batch_size] for i in range(0, train_size, batch_size)]
        batches_y = [label_data[:,i: i+batch_size] for i in range(0, train_size, batch_size)]
        batch_len=len(batches_x)
        loss=np.array([])
        losses=np.array([])
        accuracy=np.array([])
        for item in range(self.epoch):
            print('This is %d epochs' % item)
            for iteration in range(batch_len):

                z1, a1 = self.forward_hiddenlayer(batches_x[iteration], self.w1, self.b1)
                z2, a2 = self.forward_outlayer(a1, self.w2, self.b2)

                # 反向传播过程
                dz2=self.back_outlayer(a2,batches_y[iteration],a1,batch_size,self.w2)
                dz1=self.back_hiddenlay(dz2,a1,batches_x[iteration],batch_size,self.w1)
                #loss=np.append(loss,self.crossEntropy(a2,batches_y[iteration]),batch_size)
                #print(loss)
                loss=np.append(loss,self.crossEntropy_L2(a2,batches_y[iteration],batch_size,self.w1,self.w2))
                if(iteration%50==0):
                    losses=np.append(losses,np.mean(loss))
                    loss=np.array([])

            accuracy=np.append(accuracy,self.predict(x_test,y_test))
        return losses,accuracy       
    # -----------------预测----------------------------
    def predict(self, input_data, label):
        precision = 0
        for i in range(10000):
            z1, a1 = self.forward_hiddenlayer(input_data[:, i].reshape(-1, 1), self.w1, self.b1)
            z2, a2 = self.forward_outlayer(a1, self.w2, self.b2)
            #print("a2:",a2)
            #print('模型预测值为:{0},\n实际值为{1}'.format(np.argmax(a2), label[i]))
            #print("max:",np.argmax(a2))
            if np.argmax(a2) == label[i]:
                precision += 1
        accuracy=100 * precision / 10000
        print("accuracy:%d" % (100 * precision / 10000) + "%")
        return accuracy

if __name__ == '__main__':
    # 输入层数据维度784，隐藏层100，输出层10
    dl = Nerual_Network(784, 200, 10, 0.2)
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()
    # 循环训练方法
    # x1_train=x_train[:,0:10000]
    # y1_train=y_train[:,0:10000]
    # x1_test=x_test[:,0:1500]
    
    batch_size=128
    train_size=60000
    # losses=np.array([])
    losses,accuracy=dl.train(x_train, y_train,train_size,batch_size)

    print(dl.learningrate,batch_size)

    # 预测模型
    dl.predict(x_test,y_test)

    #绘制损失曲线和正确率曲线
    import matplotlib.pyplot as plt

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    #plt.plot(accuracy)
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy Curve')

    # 显示图表
    plt.show()