from scipy.optimize import  root

# optimize.root(fun, x0, options, …)
# options是较少用的参数参数fun是要解的函数名称，x0是初始迭代值（可以用不同的参数值，会有不同的结果）

def f(x):
    return (a*x**2 + b*x + c)

a = 3
b = 5
c = 1
r1 = root(f, 0)
print(r1.x)
r2 = root(f, -1)
print(r2.x)

# 解线性方程组
def fun(x):
    return (a*x[0]+b*x[1]+c, d*x[0]+e*x[1]+f)
a = 2
b = 3
c = -13
d = 1
e = -2
f = 4
r = root(fun, [0, 0])
print(r.x)

# 计算2个线性方程的交点
import matplotlib.pyplot as plt
import  numpy as np
def fx(x):
    return (x**2 - 2)

def fy(x):
    return (-x**2 + 2*x +2)

r1 = root(lambda x: fx(x)-fy(x), 0)  # 初始迭代值0，可以理解为初步猜测的根
r2 = root(lambda x: fx(x)-fy(x), 5)  # 初始迭代值5，可以理解为初步猜测的根
# print(r1)
# print(r2)
print("x1=%4.2f, y1=%4.2f"%(r1.x, fx(r1.x)))
print("x2=%4.2f, y2=%4.2f"%(r2.x, fx(r2.x)))
