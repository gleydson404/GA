import numpy as np
from MLPXORGA import feed_forward
inputs = np.matrix([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]])
RIGHT_ANSWER = [0, 1, 1, 0]

# chromosome = np.matrix([-3.20601877, -16.79708469, -16.56907027, -15.70006945, -11.85963456, -10.14149513,   4.89624683, -15.4298402 ,  11.32024019])
chromosome = np.matrix([[-9.8050,   -6.0907,   -7.0623], [-2.4839,   -5.3249,   -6.9537], [5.7278,   12.1571,  -12.8941]])
matrix = chromosome.reshape(3, 3)

w = np.matrix(matrix[:2, :])
z = np.matrix(matrix[2, :])

for input in inputs:
    print feed_forward(input, w.T, z.T)