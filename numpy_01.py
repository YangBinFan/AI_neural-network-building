
import numpy as np

##A = np.array([1, 2, 3, 4])
##print(A)
##
##print(np.ndim(A))
##print(A.shape)
##
##B = np.array([[1, 2], [3, 4], [5, 6]])
##print(np.ndim(B))
##print(B.shape)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[1, 2, 3], [4, 5, 6]])

#print(np.dot(A, B))
print('A_dim=', A.shape)
print('C_dim=', C.shape)

print(np.dot(A, C))
print(np.dot(C, A))
