import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 101)
y = [y * y for y in x]
plt.axis([-5, 5, 0, 30])
plt.plot(x, y)
plt.xticks(range(-5, 6, 1))
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()