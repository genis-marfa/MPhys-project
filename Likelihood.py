# Likelihood
import sys
import pickle
import numpy             as np
import scipy.stats       as stats
import matplotlib.pyplot as plt
inputs = pickle.load(open("paper_values_and_DMEFT_D7a_m10_l400_prediction.pickle", "rb"))

meas         = inputs["meas_values"]              #  This is the measurement (central values)
meas_cov     = inputs["meas_cov"]                 #  This is the measurement (covariance)
meas_cov_inv = np.linalg.inv(meas_cov)            #  This is the inverse covariance
n_dof        = len(meas)                          #  This is the number of bins (=24)

SM           = inputs["SM_model_values"]          #  This is the SM expectation (central values)
SM_exp_cov   = inputs["SM_exp_cov"]               #  This is the expected experimental covariance
SM_thry_cov  = inputs["SM_model_cov"]             #  This is the covariance on the central values due to theory modelling
SM_thry_cov  = np.zeros(shape=(n_dof, n_dof))               # Uncomment this line if you want to ignore theory uncertainty on SM
SM_total_cov = SM_exp_cov + SM_thry_cov           #  This is the sum of the theory covariance with the expected experimental covariance
SM_exp_cov_inv   = np.linalg.inv(SM_exp_cov)      
SM_total_cov_inv = np.linalg.inv(SM_total_cov)

# MadGraph Generated Data goes here:
BSM_400     = inputs["BSM400_values"]             #  This is Ste's MG5 (LO) expectation for DMEFT D7a: m_chi = 10 GeV, lambda = 400 GeV
BSM_400_cov = inputs["BSM400_cov"]                #  This is the covariance on the expectation
BSM_400_cov = np.zeros(shape=(n_dof, n_dof))      
#  ============================================================================
#      G: LOAD OWN MG5 DATA (PICKLE):
#  ============================================================================
with open('MG_Data_m100.pickle', 'rb') as f:
    inputs = pickle.load(f)

mchi        = inputs[0]    # mchi used
lda         = inputs[1]    # Lambda used 
Rmiss_SM    = inputs[2]    # MG5(LO) Rmiss for SM prediction
Rmiss_DM    = inputs[3]    # MG5(LO) Rmiss for DM only prediction.
cov_stat_SM = inputs[4]    # Statistical Covariance Matrix for SM prediction
cov_stat_DM = inputs[5]    # Statistical Covariance Matrix for DM prediction

Rmiss_SM=np.asarray(Rmiss_SM)
Rmiss_DM=np.asarray(Rmiss_DM)
Rmiss_BSM=Rmiss_SM+Rmiss_DM  


cov_SM=cov_stat_SM+SM_exp_cov;
#  Experimental covariance (cov_stat_SM) + theory covariance (MC modelling)

cov_DM=cov_stat_DM;
MG_SM_cov_thry=cov_stat_SM


def get_chi2 (meas, pred, cov) :
	cov_inv = np.linalg.inv(cov) # Computes covariance matrix inverse.
	res = meas - pred; 
	return np.matmul(res, np.matmul(cov_inv, res))


def pred(lda):
    return SM+Rmiss_DM*(400/lda)**6
    

det=np.linalg.det(meas_cov)
normal_factor = 1.0/( (2*np.pi)**n_dof *det)

lda_space = np.linspace(500,1000,100)
LogLikelihood=[];
for lda in lda_space:
    L=normal_factor*np.exp(-0.5*get_chi2(meas, pred(lda), meas_cov))
    LL=np.log(L)
    LogLikelihood.append(LL)

plt.plot(lda_space, LogLikelihood)

plt.plot(np.ones(100)*875.5, LogLikelihood )

plt.close()

DM1 = Rmiss_DM*(400)**6
lda_space = np.linspace(500,2000,100)
TEST=[]
for lda in lda_space:
    K=6* (pred(lda)-meas) *DM1;
    W=(np.linalg.norm(K))
    TEST.append(W)
plt.plot(lda_space, TEST, 'r')


plt.plot(lda_space, np.zeros(100), 'g')

