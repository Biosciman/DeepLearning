import matplotlib.pyplot as plt
alcohol = 58
for x in range(0, 11):
    if x > 0:
        alcohol *= 0.5
        print(f"当x={x:2d}，酒精读书={alcohol}")
        print("当x={0}，酒精读书={1}".format(x, alcohol))

alcohol = 58
x = [x for x in range(0,11)]
y = [alcohol * (1/2) ** y for y in x]
plt.axis([0, 12, 0, 60])
plt.plot(x, y)
plt.xlabel('Times')
plt.ylabel('Alcohol concentration')
plt.grid()
plt.show()