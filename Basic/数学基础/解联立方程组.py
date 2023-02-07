from sympy import solve, symbols

a, b = symbols('a, b')
eq1 = 28*a + 12*b -170
eq2 = 12*a + 6*b -70
ans = solve((eq1, eq2))
print(ans)