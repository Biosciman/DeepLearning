import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.integrate import trapz
import numpy as np

x = np.linspace(-3, 3, 100)
y = st.norm.pdf(x)  # 正态分布
plt.plot(x, y)

xs = np.linspace(-0.8, -0.6, 100)
p = trapz(st.norm.pdf(xs), xs)  # trapz计算积分
print((f"落在-0.8与——0.6之间的概率是{100*p:4.2f}%"))
plt.fill_between(xs, st.norm.pdf(xs), color='green')

plt.grid()
plt.show()