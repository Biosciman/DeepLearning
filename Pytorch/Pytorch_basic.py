# -*- codeing = utf-8 -*-
# 参考https://zhongqiang.blog.csdn.net/article/details/104506659

import torch
import numpy as np

# torch构建Tensor
# 构造一个未初始化的5 * 3的矩阵
x = torch.empty(5, 3)    # 5行3列的未初始化Tensor

# 构造一个随机初始化的矩阵
x = torch.rand(5, 3)  # 5行3列的随机初始化Tensor， 都在0-1之间

# 构造一个全部为0， 类型为long的矩阵
x = torch.zeros(5, 3, dtype=torch.long)   # dtype属性可以指定类型,5行3列的全0的tensor

# 从数据中直接构造tensor
x = torch.tensor([5.5, 3])   # tensor([5.5000, 3.0000])

# 可以从一个已有的Tensor构建一个tensor
# 这些方法会重用原来的Tensor的特征，例如数据类型，除非提供新的数据
x = x.new_ones(5, 3, dtype=torch.double)  # 5行3列全1Tensor
y = torch.randn_like(x, dtype=torch.float)  # y也是5行3列

# 得到Tensor的形状
x.size()

# 简单的运算
# 加法运算
x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
y = torch.rand(5, 3)
# print(x)
# print(y)
# print(x+y)
# print(torch.add(x, y))  # 另一种加法的写法

# 加法： 把输出作为一个变量
result = torch.empty(5, 3)
torch.add(x, y, out=result)
# print(result)

# in-place加法
#  任何in-place的运算都会以``_``结尾。举例来说：``x.copy_(y)``, ``x.t_()``, 会改变 ``x``
y.add_(x)
# print(y)   # 这时候是y本身加上了x

# 各种类似NumPy的indexing都可以在PyTorch tensor上面使用
# 切片
# print(x[:, 0:2])

# Resizing:如果希望resize/reshape一个tensor，可以使用torch.view
x = torch.randn(4, 4)
y = x.view(8, 2)   # 相当与成了8行2列
# print(x)
# print(y)

# 如果有一个只有一个元素的tensor，使用.item()方法可以把里面的value变成Python数值
x = torch.randn(1)
# print(x)
# print(x.item())

# Numpy和Tensor之间的转化
# Torch Tensor和Numpy array会共享内存，所以改变其中一项也会改变另一项
# Torch Tensor 转变成Numpy array
a = torch.ones(5)
# print(a)
b = a.numpy()
# print(b)

# 改变tensor,也会改变numpy里面的数值
a.add_(1)
# print(a)
# print(b)

# Numpy array转成Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
# print(a)
# print(b)

np.add(a, 1, out=a)  # 这个是在原内存上操作
# print(a)
# print(b)

a = a + 1   # numpy变了，tensor不变
# print(a)
# print(b)

# 用numpy实现两层神经网络
# 定义样本数，输入层，隐藏层，输出层的参数
# 每个样本有1000个变量，输入层的神经元数量=变量数量
N, D_in, H, D_out = 64, 1000, 100, 10

# 创造训练样本x,y  这里随机产生
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
# print(x)
# print(y)

# 随机初始化参数w1, w2
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
# print(w1)

# 下面就是实现神经网络的计算过程
learning_rate = 1e-6
epochs = 500

for epoch in range(epochs):
    # 前向传播
    # dot()返回的是两个数组的点积,即w1和x。点积是指在向量的每个位置上的元素的乘积之和。形成一个64*100的矩阵
    h = x.dot(w1)  # 训练样本x输入输入层w1，输入层输出参数h。
    # maximum()函数用于查找数组元素的逐元素最大值,h中的元素与0比较，取最大值即删除负值。形成一个64*100的矩阵
    # ReLu会使一部分神经元的输出（小于0部分）为0，这样就造成了 网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。
    h_relu = np.maximum(h, 0)  # 参数h输入激发函数，得到h_relu。
    # 形成一个64*10的矩阵
    y_pred = h_relu.dot(w2)  # h_relu输入输出层，得到结果。

    # 计算损失
    # square获得矩阵平方，sum将矩阵中所有元素相加
    loss = np.square(y_pred - y).sum()
    print(epoch, loss)  # loss会越来越小

    # 反向传播
    # w2的梯度
    grad_y_pred = 2.0 * (y_pred - y)  # 形成一个64*10的矩阵
    # T为转置，形成100*64的矩阵
    grad_w2 = h_relu.T.dot(grad_y_pred)  # 形成一个100*10的矩阵
    # w1的梯度
    grad_h_relu = grad_y_pred.dot(w2.T)  # 形成一个64*100的矩阵
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0  # 将矩阵中小于0的元素归零，去掉负值
    grad_w1 = x.T.dot(grad_h)  # 形成一个1000*100的矩阵

    # 更新参数
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2


