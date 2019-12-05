
# Numpy 安裝
# pip install numpy

# Numpy

# ex01
##import numpy
##import numpy as np
##
##x = np.array([1.0, 2.0, 3.0])
##print(x)
##print(type(x))

# ex02
##import numpy as np
##
##x = np.array([1.0, 2.0, 3.0])
##y = np.array([2.0, 4.0, 6.0])
##print(x+y)
##print(x-y)
##print(x*y)  # element-wise product
##print(x/y)

# ex03
##import numpy as np
##
##A = np.array([[1, 2], [3, 4]])
##print(A)
##print(A.shape)
##print(A.dtype)
##
##B = np.array([[3, 0], [0, 6]])
##print(A + B)
##print(A * B)

# ex04 廣播 broadcast
##import numpy as np
##
##A = np.array([[1, 2], [3, 4]])
##B = np.array([10, 20])
##C = np.transpose(B)
##print(C)
##print('B.shape=', B.shape)
##print('C.shape=', C.shape)
##
##D = B.reshape(B.shape[0],1)
##print(D)
##
##print('A=', A)
##print(A * B)
##print(A * D)

# ex05 存取List元素

import numpy as np

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X.shape)

print(X[0])
print(X[0][1])

X = X.flatten() # 轉換為一維陣列
print(X)

print(X[np.array([0, 2, 4])]) # 取索引 0, 2, 4


















