from sympy import *

x = Symbol('x')
f = 58 * (1 / 2) ** x
print("result = ", limit(f, x, float('inf')))  # float('inf')为计算x趋近无限大表达方式
print('result = ', limit(f, x, oo))  # oo为计算x趋近无限大表达方式

x = Symbol('x')
f1 = 1 / x
print("右边趋近 0 = ", limit(f1, x, 0))  # 0为计算x趋近0的表达方式
print("左边趋近 0 = ", limit(f1, x, 0, dir='-'))  # dir='-'表示从左边趋近