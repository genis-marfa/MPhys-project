# Get CL limits:
import sys
import matplotlib.pyplot as plt
import numpy             as np
import scipy.stats       as stats
import pickle
#  ====================================================================================================
#      GLOBAL: READ DATA, SM AND BSM PREDICTIONS:
#  ====================================================================================================
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
#      G: SCAN THROUGH DIFFERENT MASSES:
#  ============================================================================
f = pickle.load(open("DM_scan_points.pickle","rb"))

BSM_400_allmasses=[];
masses=[];
for item in f: 
    mass=item[0]
    masses.append(mass)
    BSM_400_allmasses.append(item[2])

masses=masses[0:(len(masses)-1)]
BSM_400_allmasses=BSM_400_allmasses[0:(len(BSM_400_allmasses)-1)]

#  ============================================================================
#      G: LOAD OWN MG5 DATA (PICKLE):
#  ============================================================================
with open('MG_Data_m900.pickle', 'rb') as f:
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


SM_thry   =[];
SM_thry = np.zeros(shape=(n_dof, n_dof))  

if True:
# TESTS FOR SM THEORY ERRORS:
    k = [];
    k[:]=[0.005 for i in range (24)]
    SM_thry   = np.diag(k) 

cov_DM=cov_stat_DM;
MG_SM_cov_thry=cov_stat_SM
# Theory covariances (from MC modelling). These are the statistical covariances
# from the MG simulution. (Diagonal matrices of variances.) 


# Set this to true to print comparison HepData-MG simulated data.
data_compare=False;

if data_compare:    
    print('Mass used:', mchi)
    print('SM HepData prediction:')
    print('----------------------')
    print(SM)
    print('')
    print('SM MG(LO) Data prediction:')
    print('-------------------------')
    print(np.around(Rmiss_SM, decimals=5))
    print('')
    print("HepData EFT DM prediction at mass",str(masses[3]),'lambda 400:')
    print('------------------------------------------------')
    print(BSM_400_allmasses[3])
    print('')
    print('Madgraph EFT DM at mass ', mchi,' lambda ', lda,':')
    print('---------------------------------------------')
    print(np.around(SM+Rmiss_DM, decimals=6))
    print('')

#  ====================================================================================================
#      REQUIRED FUNCTIONS:
#  ====================================================================================================
#  Calculate chi2 between two distributions with a specified covariance:
def get_chi2 (meas, pred, cov) :
	cov_inv = np.linalg.inv(cov) # Computes covariance matrix inverse.
	res = meas - pred; 
	return np.matmul(res, np.matmul(cov_inv, res))

#  Calculate the confidence level (p-value) of a chi2 test-statistic assuming that it follows the usual distribution
def get_frequentist_CL (chi2) :
	return 1.0-stats.chi2.cdf(chi2, n_dof) #cdf: Cummulative Distribution Function
# Stats paper: Practical stats for particle physics, Barlow: p. 40, point 4: Upper Limit. 

#  Calculate the CLs limit associated with a given measurement
#  - method: profile lambda, create a BSM prediction at each lambda value, add it to the SM and calculate CLs. Return value of lambda where profile crosses (1-coverage)

def get_CLs_limit(this_meas, this_cov, this_mass, coverage=0.95):
    if coverage <= 0 or coverage >= 1 : raise ValueError("get_CLs_limit(): provided coverage {coverage} is out of the allowed range [0, 1]")
    lambda_linspace=np.linspace(500, 1010, 51);
    CLs=[]; 
 
    for this_lambda in lambda_linspace: 
        scale_factor=(400./this_lambda) ** 6;
        BSM=BSM_400_allmasses[this_mass] * scale_factor;
        BSM_cov= BSM_400_cov * scale_factor * scale_factor;
        chi2_BSM=get_chi2(this_meas, SM + BSM, this_cov + SM_thry_cov + BSM_cov);
        chi2_SM= get_chi2(this_meas, SM, this_cov + SM_thry_cov)
        CLs.append( get_frequentist_CL(chi2_BSM) / get_frequentist_CL(chi2_SM) );
    this_CLs=np.interp([1.0-coverage], CLs, lambda_linspace)[0];
    return(this_CLs)
    
def get_CLs_limit_MG (this_meas, this_cov, coverage=0.95):
    if coverage <= 0 or coverage >= 1 : raise ValueError("get_CLs_limit(): provided coverage {coverage} is out of the allowed range [0, 1]")
    lambda_linspace=np.linspace(500, 2000, 101);
    CLs=[]; 
    
    for this_lambda in lambda_linspace: 
     #   print('Lambda:', this_lambda)
        scale_factor=(400./this_lambda) ** 6;
        RmissDM = Rmiss_DM * scale_factor;
     #   print('Res:', this_meas-RmissDM-Rmiss_SM)
        cov_DM=cov_stat_DM * scale_factor * scale_factor;
        chi2_DM= get_chi2(this_meas, Rmiss_SM + RmissDM, this_cov + cov_DM + cov_stat_SM+SM_thry);
        #   print('chi2_DM:',chi2_DM)
        chi2_SM= get_chi2(this_meas, Rmiss_SM, this_cov + MG_SM_cov_thry)
#        print('chi2_SM:',chi2_SM)
        CLs.append( get_frequentist_CL(chi2_DM) / get_frequentist_CL(chi2_SM) );
    this_CLs=np.interp([1.0-coverage], CLs, lambda_linspace)[0];
    return(this_CLs)

def remove_bin(plot, n): # nth bin in specific plot (starting at 0th bin)
    n=n-1
    if plot=='b':
        n+=7
    if plot=='c':
        n+=13
    if plot=='d':
        n+=18

    global cov_stat_SM, cov_stat_DM, MG_SM_cov_thry, meas_cov, SM_exp_cov, Rmiss_DM, Rmiss_SM, SM, meas;

    # Remove corresponding row & column from covariance matrices:
    cov_stat_SM=np.delete(cov_stat_SM, n, axis=0)
    cov_stat_SM=np.delete(cov_stat_SM, n, axis=1)
    
    cov_stat_DM=np.delete(cov_stat_DM, n, axis=0)
    cov_stat_DM=np.delete(cov_stat_DM, n, axis=1)
    
    MG_SM_cov_thry=np.delete(MG_SM_cov_thry, n, axis=0)
    MG_SM_cov_thry=np.delete(MG_SM_cov_thry, n, axis=1)
    
    meas_cov=np.delete(meas_cov, n, axis=0)
    meas_cov=np.delete(meas_cov, n, axis=1)
    
    SM_exp_cov=np.delete(SM_exp_cov, n, axis=0)
    SM_exp_cov=np.delete(SM_exp_cov, n, axis=1)
    
    # Remove entries in data set:
    Rmiss_DM=np.delete(Rmiss_DM,n)
    Rmiss_SM=np.delete(Rmiss_SM,n)
    SM=np.delete(SM,n)
    meas=np.delete(meas,n)
    
    if plot=='a':
        print('Removed bin '+str(n+1)+' from plot a')    
    if plot=='b':
        print('Removed bin '+str(n+1-7)+' from plot b')
    if plot=='c':
        print('Removed bin '+str(n+1-13)+' from plot c')
    if plot=='d':
        print('Removed bin '+str(n+1-18)+' from plot d')
    
# Diagonalise covariance matrices: Remove correlations from other bins. 
def diagonalise_meas():
    global meas_cov;
    diags    = np.diag(meas_cov)
    meas_cov = np.diag(diags);
    print('Removed off-diagonal entries in measurement covariance')

def diagonalise_SM_exp():
    global SM_exp_cov
    diags      = np.diag(SM_exp_cov);
    SM_exp_cov = np.diag(diags);
    print('Removed off-diagonal entries in SM expected covariance')

#  ============================================================================
#      MAIN SCRIPT:
#  ============================================================================

print('CLs Limits BEFORE:')
print('-----------------')
print('Own MG5 simulation:')
print('Observed Limit at mass', mchi, ':', get_CLs_limit_MG(meas,     meas_cov))
print('Expected Limit at mass', mchi, ':', get_CLs_limit_MG(Rmiss_SM, SM_exp_cov))
print('')
print("Stephen's Code:")
print('Observed Limit at mass:', masses[3], ':', get_CLs_limit(meas, meas_cov,3))
print('Expected Limit at mass:', masses[3], ':', get_CLs_limit(SM, SM_exp_cov,3))
print('_____________________________________________________')
print('')
print('')

# Remove particular bin, and corresponding row & column in all cov. matrices:
# To call function: use plot 'a', 'b', 'c', or 'd', and the particular bin for that plot: 
# n = {1,...,7} for 'a'.(7 bins total)
# n = {1,...,6} for 'b' and 'd'. (6 bins total)
# n = {1,...,5} for 'c'. (5 bins total)
# To call function, use n=0,...,6/5/4/5. for a,b,c,d respectively. 
# n = 1,...,24 for plots a, b, c, d

print('Modifications:')
print('--------------')
# When removing bins, make sure to do so from latest to first. Otherwise mess up order of n. 
# remove_bin('d',1)
#remove_bin('c',5)
#remove_bin('b',6) 
#remove_bin('b',5)
#remove_bin('b',3)
#remove_bin('a',6)

# Uncomment to remove off-diagonal entries of covariance matrix: Get rid of bin correlations. 
# diagonalise_meas()
# diagonalise_SM_exp()


print('')
print('')

print('CLs Limits AFTER:')
print('-----------------')
print('Own MG5 simulation:')
print('Observed Limit at mass', mchi, ':', get_CLs_limit_MG(meas,     meas_cov))
print('Expected Limit at mass', mchi, ':', get_CLs_limit_MG(Rmiss_SM, SM_exp_cov))
print('_____________________________________________________')




