
import numpy as np
import scipy.io

data = {}

variables = scipy.io.loadmat( "mult_nor.mat")
data["content"] = variables['X']
print(variables['X'].dtype)
print(variables['X'].shape)
