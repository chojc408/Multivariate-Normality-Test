import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import normaltest
from scipy.stats import chi2
from scipy.stats import norm

"""
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

(Note that, if it is an univariate normality test,
 each vector has only one element.)

All calculations are done with vertical vectors
(shape = (k, 1))
=======================================================
"""

def get_example_multi_normal(k=4, n=100):
    # will generate n-by-k matrix
    # mean = 0, std = 1
    # assumes independent variables (dimensions)
    mean = np.zeros(k) 
    cov  = np.eye(k)
    X = np.random.multivariate_normal(mean, cov, size=n)
    return X

def get_example_gamma(alpha=1, n=100):
    column_1 = np.random.gamma(alpha, size=n)
    column_2 = np.random.gamma(alpha, size=n)
    column_3 = np.random.gamma(alpha, size=n)
    column_4 = np.random.gamma(alpha, size=n)
    X = np.vstack((column_1, column_2, column_3, column_4))
    X = X.T
    return X

def Mardia_MVN_test(X, verbose=1):
    print("==========================================")
    print("Mardia's Multivariate Normality Test")
    print("------------------------------------------")
    print("Do NOT use this test for large sample")
    print("computation time ~ num_SamplePoint**2")
    print("------------------------------------------")
    n = X.shape[0]
    k = X.shape[1]
    print("the number of dimesions    :", k)
    print("the number of sample points:", n)
    print("------------------------------------------")
    # === x_avg ==============================   
    x_avg = (1/n)*np.sum(X, axis=0)
    x_avg = x_avg.reshape((k, 1))
    # === inverse of sample_covariance_matrix
    S = np.cov(X.T)
    # for numpy function np.cov(M),
    # each row of M represents a variable (dimension)
    # each column of M represent a single observation
    # thus, transpose required
    S_inv = np.linalg.inv(S) # inverse of S
    # === b1 and b2 ========================== 
    sum_for_b1 = 0
    sum_for_b2 = 0
    for i in range(n):
        if verbose == 1:
            if i%100 == 0:
                print("calcultion", int((i/n)*100), "% done")
        x_i = X[i].reshape((k,1))
        delta_i = x_i - x_avg
        dot_p_i = np.dot(delta_i.T, S_inv)
        g_ii = np.dot(dot_p_i, delta_i) # g_ii
        sum_for_b2 = sum_for_b2 + g_ii**2
        for j in range(n):
            x_j = X[j].reshape((k,1))
            delta_j = x_j - x_avg
            g_ij = np.dot(dot_p_i, delta_j) # g_ij
            sum_for_b1 = sum_for_b1 + g_ij**3
    b1 = sum_for_b1 / n**2 
    b2 = sum_for_b2 / n
    b1 = b1[0][0] # array to number. mSkewness
    b2 = b2[0][0] # array to number. mKurtosis
    # === z1 and z2
    z1 = (((k+1)*(n+1)*(n+3))/(6*((n+1)*(k+1)-6)))*b1
    z2 = (b2-k*(k+2))/((8*k*(k+2)/n)**0.5)
    # === chi-square probability
    chi2_df = k*(k+1)*(k+2)/6
    p_chi2 = chi2.cdf(z1, chi2_df)
    # === normal probability
    p_norm = 1- (1-norm.cdf(abs(z2), 0, 1))*2
    # === results
    print("------------------------------------------")
    print("mSkewnes  (b1):", np.round(b1, 1),
          "chi-2  p:", 1-np.round(p_chi2,4))
    print("mKurtosis (b2):", np.round(b2, 1),
          "normal p:", 1-np.round(p_norm,4))
    print("==========================================")
    return None

def Doornik_Hansen_MVN_test(X, verbose=1):
    print("==========================================")
    print("Doornik-Hansen Multivariate Normality Test")
    print("------------------------------------------")
    n = X.shape[0]
    k = X.shape[1]
    print("the number of dimesions    :", k)
    print("the number of sample points:", n)
    print("------------------------------------------")
    # === x_avg =======================================   
    x_avg = (1/n)*np.sum(X, axis=0)
    x_avg = x_avg.reshape((k, 1))
    # === inverse of sample_covariance_matrix =========
    S = np.cov(X.T)
    # for numpy function np.cov(M),
    # each row of M represents a variable (dimension)
    # each column of M represent a single observation
    # thus, transpose required
    S_inv = np.linalg.inv(S) # inverse of S
    # === V and C = VSV (correlation matrix) ========== 
    D = np.sqrt(np.diag(S))
    D = np.diagflat(D)
    V = np.linalg.inv(D)
    C = np.dot(np.dot(V, S), V) # correlation matrix
    # === eigenvectors and eignevalues of C
    eig_val, eig_vec = np.linalg.eig(C)
    L = np.diagflat(eig_val**-0.5)
    H = eig_vec.T       # columns are eigenvectors
    # === transformed X
    X_center = X - x_avg.T
    dot_prod = np.dot(X_center, V)
    dot_prod = np.dot(dot_prod, H)
    dot_prod = np.dot(dot_prod, L)
    X_transf = np.dot(dot_prod, H.T)
    # === skewness and kurtosis of transformed X
    skewn_list = [ ]
    kurto_list = [ ]
    for dimension in range(k):
        x = X_transf[:, dimension]
        skewn = skew(x)
        kurto = kurtosis(x, fisher=False) # Pearson definition
        skewn_list.append(skewn)
        kurto_list.append(kurto)
    # === z1 and z2 ==================================
    z1_list = [ ]
    z2_list = [ ]
    for p in range(k):
        b1     = (skewn_list[p])**2
        b2     = kurto_list[p]
        beta   = (3*(n**2 + 27*n - 70)*(n+1)*(n+3))/((n-2)*(n+5)*(n+7)*(n+9))
        ohm_2  = -1 + (2*(beta-1))**0.5
        delta1 = (np.log10(ohm_2**0.5))**(-0.5)
        y      = ((b1*(ohm_2 - 1)*(n+1)*(n+3))/(12*(n-2)))**(0.5)
        z1     = delta1*(np.log10(y+ (1+y**2)**0.5))
        z1_list.append(z1)
        # -------------------------------------------------------------
        delta2 = (n-3)*(n+1)*(n**2+15*n-4)
        a      = ((n-2)*(n+5)*(n+7)*(n**2+27*n-70))/(6*delta2)
        c      = ((n-7)*(n+5)*(n+7)*(n**2+2*n-5))/(6*delta2)
        f      = ((n+5)*(n+7)*(n**3+37*(n**2)+11*n-313))/(12*delta2)
        alpha  = a + b1*c
        chi    = 2*f*(b2 - 1 - b1)
        z2     = ((9*alpha)**0.5)*((chi/(2*alpha))**(1/3) -1 + (1/(9*alpha)))
        z2_list.append(z2)
    Z1 = np.array(z1_list).reshape(k,1) # vertical vector
    Z2 = np.array(z2_list).reshape(k,1) # vertical vector
    # === chi2 prob ==================================
    statistic = (np.dot(Z1.T, Z2) + np.dot(Z2.T, Z2)) #[0][0]
    chi2_df = 2*k
    p_chi2 = 1- chi2.cdf(statistic, chi2_df)
    print("statistic: %.4f" % statistic)
    print("p-val    : %.4f" % p_chi2)
    print("------------------------------------------")
    print("If p-val < alpha (0.05), reject H0.")
    print("Note that H0 is multi-normality,")
    print("larger p-val indicate multi-normality")
    print("==========================================")
    return None

def UVN_test(X):
    n = X.shape[0]
    k = X.shape[1]
    for p in range(k):
        data  = X[:, p]
        mean  = np.mean(data)
        stdev = np.std(data)
        skewn = skew(data)
        kurto = kurtosis(data)
        results = normaltest(data)
        print("Dim", p, "| m: %.2f" % mean, "s: %.2f" % stdev,
                        "sk: %.2f" % skewn, "ku: %.2f" % kurto,
                        "--- p-val: %.4f" % results[1])

def MVN_test(X):
    # Use Doornik_Hansen_MVN_test for large samples.
    # It is more conservative than Mardia's test,
    # but fast calculation.
    Doornik_Hansen_MVN_test(X)
    UVN_test(X)
    return None

# === Example dataset
X = get_example_multi_normal(k=16, n=100)
#X = get_example_gamma(alpha=1, n=100)

# === Main
MVN_test(X)
