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
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

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

train_load=DataLoader(dataset=train_data,batch_size=1000,shuffle=True)
test_load=DataLoader(dataset=test_data,batch_size=500,shuffle=True)
train_x,train_y=next(iter(train_load))

print(type(train_load))

# ------------------梯度------------------------------






# -----------------softmax函数-------------------------
#参数：X为一个向量
def softmax(X):
    X_exp=torch.exp(X)
    exp_sum=X_exp.sum(dim=1,keepdim=True) 
    return X_exp/exp_sum

# -----------------前向传播----------------------------




# -----------------反向传播----------------------------




# -----------------交叉熵损失函数-----------------------




# -----------------训练模型----------------------------





# -----------------预测----------------------------