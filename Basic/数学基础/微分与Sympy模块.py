from sympy import *

x = Symbol('x')
f = Symbol('f')
f = 3 * x ** 2 + 2 * x + 10
print("f'(x)=", f.diff())
print("f'(x)=", diff(f, x))

# 二次微分
print("f''(x)=", diff(f, x, 2))