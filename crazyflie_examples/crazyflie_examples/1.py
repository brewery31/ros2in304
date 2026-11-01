from itertools import combinations, product
from random import choice
import numpy as np
Z = 0.7
vertices = list(product([2, -2], [0], [0.7, 1.5]))+list(product([0],[2, -2], [0.7, 1.5]))
vertices = list(product([1.5,-1.5], [1.5,-1.5], [Z, 1.5]))
temp = 1
vertices = list(product([temp, -temp], [temp, -temp], [Z, 1.4]))
print(vertices[3])

a = np.array([-1,1])
print(list(range(len(vertices))))
# vertices = [np.array(j+(0.5,)) for j in vertices]
print(vertices)