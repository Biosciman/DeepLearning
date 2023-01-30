from sympy import *

x = Symbol('x')
a = Symbol('a')
f = a * x
print(integrate(f, x))

# 使用Sympy计算定积分的值
x = Symbol('x')
f = x ** 2
print(integrate(f, (x, 1, 3)))