import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 101)
y = [y * y for y in x]
plt.axis([-2, 2, 0, 4])
plt.plot(x, y)
plt.plot([1, 2], [1, 4], '-o')
plt.xticks(range(-2, 3, 1))
plt.text(1-0.15, 1+0.1, 'A')
plt.text(2-0.15, 4-0.15, 'B')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()