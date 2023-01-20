import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, -0.01, 101)
print(x)
y = [1 / y for y in x]
print(y)
plt.axis([-1, 1, -101, 101])
plt.plot(x, y)
plt.plot(x[100], y[100], '-o')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()