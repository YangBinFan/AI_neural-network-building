
# 均方誤差 mean squared error

##import numpy as np
##
##def mean_squared_error(y, t):
##    return 0.5 * np.sum((y - t)**2)
##
##t = [0,0,1,0,0,0,0,0,0,0]
###y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
##
##y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
##print(mean_squared_error(np.array(y), np.array(t)))


# 交叉熵誤差 cross entropy error

import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))

t = [0,0,1,0,0,0,0,0,0,0]
#y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
