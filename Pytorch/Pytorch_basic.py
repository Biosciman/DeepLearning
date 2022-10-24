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


