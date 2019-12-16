  1 import numpy as np
  2
  3 mat_a = np.array([[2, 3, 4], [1, 2, 4]])
  4 mat_b = np.array([[3, 4], [4, 3], [2, 4]])
  5
  6 mat_c = np.dot(mat_a, mat_b)
  7
  8
  9 mat_d = np.zeros([2,2])
 10
 11
 12 # Please employ the for loops to finish the matrix multiplication.
 13 # Your answer "mat_d" should be equivalent to "mat_c"
 14
 15 for idx1 in range(2):
 16
 17     for idx2 in range(2):
 18
 19         for idx3 in range(3):
 20
 21             mat_d[idx1][idx2] += mat_a[idx1][idx3] * mat_b[idx3][idx2]
 22
 23 if (mat_d == mat_c).all:
 24     print ('"mat_d" is equivalent to "mat_c"')
 25     print (mat_d)

