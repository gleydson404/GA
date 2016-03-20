import numpy as np
from MLPXORGA import feed_forward
inputs = np.matrix([[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]])
RIGHT_ANSWER = [0, 1, 1, 0]

# chromosome = np.matrix([-3.20601877, -16.79708469, -16.56907027, -15.70006945, -11.85963456, -10.14149513,   4.89624683, -15.4298402 ,  11.32024019])
# chromosome = np.matrix([[-9.8050,   -6.0907,   -7.0623], [-2.4839,   -5.3249,   -6.9537], [5.7278,   12.1571,  -12.8941]])
# chromosome = np.matrix([ -8.92385602,  15.5327354,  -12.48754229,   7.1176717,   11.96837858, -14.78113472 , -6.16376509 ,-15.50667104 , 19.2540165 ])
# chromosome = np.matrix([  8.00584059, -15.16256009,  16.19844395,  -1.87975927,  -5.08007575,  14.67067878,  -7.76349944,  18.68712223, -18.43825631])
chromosome = np.matrix([-15.30818119, -13.98054087,  -8.86207526,  10.50102516, 17.80322   ,  11.11439029,  16.95528699,  14.54083388,  12.41157988])
matrix = chromosome.reshape(3, 3)

w = np.matrix(matrix[:2, :])
z = np.matrix(matrix[2, :])

for input in inputs:
    print np.rint(feed_forward(input, w.T, z.T))