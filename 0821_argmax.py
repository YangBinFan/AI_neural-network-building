
import numpy as np

x = np.array([[1.0, 0.8, 0.1],
              [0.3, 0.1, 0.6],
              [0.2, 0.5, 0.3],
              [0.8, 0.1, 0.1]])

y = np.argmax(x, axis=1)
print(y)
