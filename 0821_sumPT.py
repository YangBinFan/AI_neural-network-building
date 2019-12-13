

import numpy as np

y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y == t)
accuracy = np.sum(y == t)
print(accuracy)
