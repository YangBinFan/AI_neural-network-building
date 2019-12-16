import numpy as np 

mat_a = np.array([[2, 3, 4], [1, 2, 4]])
mat_b = np.array([[3, 4], [4, 3], [2, 4]])

mat_c = np.dot(mat_a, mat_b)


mat_d = np.zeros([2,2])


# Please employ the for loops to finih the matrix multiplication.
# Your answer "mat_d" should be equivalent to "mat_c" 

for idx1 in range(2):
    for idx2 in range(2):
        for idx3 in range(3):
            pass 




print (mat_d)
