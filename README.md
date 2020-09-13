# Multivariate-Normality-Test
========================================================
           VERY IMPORTANT NOTE
========================================================
If a sample is drawn from k-dimensional (k-variate)
space, the sample should be a vector of shape (k,).

each sample point i = vector x_i = [x1, x2, x3, ..., xk]

If there are n=3 independent observations (n sample
points) with k=5 variables (dimensions), then they are

sample_point_1: x_1 = [1, 2, 3, 4, 5]
sample_point_2: x_2 = [6, 7, 8, 9, 0]
sample_point_3: x_3 = [3, 4, 5, 6, 7]

Thus, input matrix (n-by-k) should be as follows

M = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 0],
              [3, 4, 5, 6, 7]])
=======================================================
