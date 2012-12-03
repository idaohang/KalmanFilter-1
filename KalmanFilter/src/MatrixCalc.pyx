import numpy as np
cimport numpy as np

DOUBLE = np.float64

ctypedef np.float64_t DOUBLE_t

def predict(np.ndarray[DOUBLE_t, ndim=2] F, np.ndarray[DOUBLE_t, ndim=2] mu, \
            np.ndarray[DOUBLE_t, ndim=2] St, np.ndarray[DOUBLE_t, ndim=2] FT, \
            np.ndarray[DOUBLE_t, ndim=2] Sx):
    """ predict current state and covariance """
    mu_hat = F.dot(mu)
    S_hat = (F.dot(St)).dot(FT) + Sx
    
    return mu_hat, S_hat

def update(np.ndarray[DOUBLE_t, ndim=2] z, np.ndarray[DOUBLE_t, ndim=2] H, \
           np.ndarray[DOUBLE_t, ndim=2] mu_hat, np.ndarray[DOUBLE_t, ndim=2] S_hat, \
           np.ndarray[DOUBLE_t, ndim=2] HT, np.ndarray[DOUBLE_t, ndim=2] Sz):
    """ update the current state and covariance at time t with Kalman update equation. """
    cdef np.ndarray[DOUBLE_t, ndim=2] z_tilda = z - H.dot(mu_hat) # observation error
    cdef np.ndarray[DOUBLE_t, ndim=2] S = (H.dot(S_hat)).dot(HT) + Sz # covariance of observation residue
    cdef np.ndarray[DOUBLE_t, ndim=2] K = (S_hat.dot(HT)).dot(np.linalg.inv(S)) # Kalman gain
    cdef np.ndarray[DOUBLE_t, ndim=2] mu = mu_hat + K.dot(z_tilda) # updated current estimation (mu)
    cdef np.ndarray[DOUBLE_t, ndim=2] St = S_hat - (K.dot(H)).dot(S_hat) # updated current error matrix
    
    return K, mu, St

cpdef np.ndarray[DOUBLE_t, ndim=2] future_prediction(np.ndarray[DOUBLE_t, ndim=2] F, np.ndarray[DOUBLE_t, ndim=2] mut, int step):
    
    cdef int i
    
    for i in xrange(step):
        mut = F.dot(mut)
        
    return mut