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
with open('MG_Data_m500.pickle', 'rb') as f:
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
# Theory covariances (from MC modelling). These are the statistical covariances
# from the MG simulution. (Diagonal matrices of variances.) 

print('Mass used:', mchi)
print('')
# Set this to true to print comparison HepData-MG simulated data.
data_compare=False;

if data_compare:    
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
	return 1.0 - stats.chi2.cdf(chi2, n_dof) #cdf: Cummulative Distribution Function
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
    this_CLs=np.interp([1.0 - coverage], CLs, lambda_linspace)[0];
    return(this_CLs)
    
def get_CLs_limit_MG (this_meas, this_cov, coverage=0.95, pred='MG'):
    if coverage <= 0 or coverage >= 1 : raise ValueError("get_CLs_limit(): provided coverage {coverage} is out of the allowed range [0, 1]")
    
    lambda_linspace=np.linspace(400, 1000, 61);
    CLs=[];    
    if pred=='MG':
        
        for this_lambda in lambda_linspace: 
            scale_factor=(400./this_lambda) ** 6;
            RmissDM = Rmiss_DM * scale_factor;
            cov_DM=cov_stat_DM * scale_factor * scale_factor;
            chi2_DM= get_chi2(this_meas, Rmiss_SM + RmissDM, this_cov+ cov_DM + MG_SM_cov_thry);
            chi2_SM= get_chi2(this_meas, Rmiss_SM, this_cov + MG_SM_cov_thry)
            CLs.append( get_frequentist_CL(chi2_DM) / get_frequentist_CL(chi2_SM) );
        this_CLs=np.interp([1.0 - coverage], CLs, lambda_linspace)[0];
        return(this_CLs) 
    
    elif pred =='HEP':
        for this_lambda in lambda_linspace: 

            scale_factor=(400./this_lambda) ** 6;
            RmissDM = Rmiss_DM * scale_factor;
            cov_DM=cov_stat_DM * scale_factor * scale_factor;
        
            chi2_DM= get_chi2(this_meas, SM + RmissDM, this_cov+ cov_DM);
            chi2_SM= get_chi2(this_meas, SM, this_cov)
        
            CLs.append( get_frequentist_CL(chi2_DM) / get_frequentist_CL(chi2_SM) );
        this_CLs=np.interp([1.0 - coverage], CLs, lambda_linspace)[0];        
        return(this_CLs) 
    
   
    
def choose_error(SM_pred, meas_data, upper_error_data, lower_error_data):
    Error=[]
    for i in range(len(meas_data)):
        if meas_data[i] > SM_pred[i]:
            this_error = lower_error_data[i]
        else:
            this_error = upper_error_data[i]
        Error.append(this_error)
    return Error

def get_chi2_perbin(meas, meas_upper_err, meas_lower_err, pred, err_pred):
    # Use "normal" chi-squared = (Observed-Expected)^2 /sigma^2
    
    chosen_meas_err=choose_error(pred, meas, meas_upper_err, meas_lower_err)
    var = [x**2+y**2 for x,y in zip(err_pred, chosen_meas_err)]
    chi2= ((meas-pred)*(meas-pred))/var;
    return chi2

meas_upper   = [0.13, 0.21, 0.27, 0.27, 0.54, 1.52, 3.64, 0.27, 0.42, 0.54, 0.43, 0.79, 2.50, 
              0.27, 0.36, 0.44, 0.78, 2.60, 0.46, 0.48, 0.45, 0.36, 0.47, 0.59]
meas_lower   = [0.13,0.1,0.14,0.13,0.28,0.64,1.4,0.27,0.21,0.28,0.22,0.42, 0.94, 
              0.27,0.19,0.24,0.41,1.18,0.46,0.25,0.22,0.19,0.25, 0.33]
err_Rmiss_SM = np.sqrt(np.diag(cov_stat_SM))
err_SM       =np.array([4.0e-02, 5.0e-02, 3.0e-02, 3.0e-02, 4.0e-02, 7.0e-02, 1.6e-01,
               8.0e-02, 9.0e-02, 5.0e-02, 5.0e-02, 7.0e-02, 1.1e-01,
               6.0e-02, 9.0e-02, 1.1e-01, 1.5e-01, 4.3e-01, 
               1.2e-01, 1.2e-01, 1.0e-01, 9.0e-02, 1.0e-01, 1.4e-01])

def get_deltachi2_perbin(this_lamda):
    chi2_MG_SM  = get_chi2_perbin(meas, meas_upper, meas_lower, Rmiss_SM, err_Rmiss_SM);  
    chi2_HEP_SM = get_chi2_perbin(meas, meas_upper, meas_lower, SM, err_SM)  
    
    BSM          = [x+y*(400/this_lamda)**6 for x,y in zip(SM, Rmiss_DM)] 
    BSM_MG       = [x+y*(400/this_lamda)**6 for x,y in zip(Rmiss_SM, Rmiss_DM)] 
    
    cov_BSM      = [x+y*(400/this_lamda)**6*(400/this_lamda)**6 for x,y in zip(SM_exp_cov, cov_stat_DM)]  
    cov_BSM_MG   = [x+y*(400/this_lamda)**6*(400/this_lamda)**6 for x,y in zip(cov_stat_SM, cov_stat_DM)]  
    
    err_BSM      = np.sqrt(np.diag(cov_BSM))
    err_BSM_MG   = np.sqrt(np.diag(cov_BSM_MG))
    
 
    chi2_HEP_BSM = get_chi2_perbin(meas, meas_upper, meas_lower, BSM, err_BSM);  
    chi2_MG_BSM  = get_chi2_perbin(meas, meas_upper, meas_lower, BSM_MG, err_BSM_MG); 

    DeltaChi2HEP = [abs(x-y) for x,y in zip(chi2_HEP_BSM,chi2_HEP_SM)]
    DeltaChi2MG =  [abs(x-y) for x,y in zip(chi2_MG_BSM, chi2_MG_SM)]

    
    return [DeltaChi2HEP, DeltaChi2MG, BSM, BSM_MG, err_BSM, err_BSM_MG]
    
    
#print('Chi2 HepData',  get_chi2_perbin(meas, meas_upper, meas_lower, SM, err_SM))
#print('Chi2 MadGraph', get_chi2_perbin(meas, meas_upper, meas_lower, Rmiss_SM, err_Rmiss_SM))
#print('Delta Chi2 MadGraph', get_deltachi2_perbin(800)[0] )
#print('Delta Chi2 HepData',  get_deltachi2_perbin(800)[1] )

#  ============================================================================
#      MAIN SCRIPT:
#  ============================================================================

print('CLs Limits:')
print('-----------')
print('Own MG5 simulation:')
#print('Observed Limit at mass', mchi, ':', get_CLs_limit_MG(meas,     meas_cov))
#print('Expected Limit at mass', mchi, ':', get_CLs_limit_MG(Rmiss_SM, SM_exp_cov))
print('')


print('HEPDATA:')
print('Observed Limit at mass', mchi, ':',  get_CLs_limit_MG(meas, meas_cov, 0.95, 'HEP'), '.')
#print('Expected Limit at mass', mchi, ':',  get_CLs_limit_MG(SM, SM_exp_cov, 0.95, 'HEP'), '.')

print('')
#print("Stephen's Code:")
#print('Observed Limit at mass:', masses[15], ':', get_CLs_limit(meas, meas_cov,15))
#print('Expected Limit at mass:', masses[15], ':', get_CLs_limit(SM, SM_exp_cov,15))
print('__________________________________________________________________________')

saveto='/Users/Genis/mphys/Rivet/DM/HNoE/plots/plots_dec_3'; 

cov_stat_HepPred=np.diag(err_SM**2)

chi2_MG= get_chi2_perbin(meas, meas_upper, meas_lower, Rmiss_SM, err_Rmiss_SM);  # MG prediction
chi2_HEP=get_chi2_perbin(meas, meas_upper, meas_lower, SM, err_SM)               # HepData prediction

OBS_CLs=[]; EXP_CLs=[];

# Mass Scan:
for mass in range(len(masses)):
    OBS_CLs.append(get_CLs_limit(meas, meas_cov, mass));
    EXP_CLs.append(get_CLs_limit(SM, SM_exp_cov, mass));

masses.remove(masses[2])
OBS_CLs.remove(OBS_CLs[2])
EXP_CLs.remove(EXP_CLs[2])


#plt.close()
folder='/Users/Genis/mphys/Rivet/DM/Calculating_CLs'; 
fig, ax2 = plt.subplots(1,figsize=(10,6),sharex=True)             
ax2.set_xlim(0, 2000)
ax2.set_ylim(10, 1000)
ax2.plot(EXP_CLs,masses,'g-', label='Exp. limit 95% CL')
ax2.plot(OBS_CLs,masses,'m-', label='Obs. limit 95% CL')
ax2.legend(loc="upper right", fontsize=14)
ax2.set_ylabel('$m_{\chi} [GeV]$', fontsize=14)
ax2.set_xlabel('EFT scale $\Lambda_{EFT}$ [GeV]',fontsize=14)
ax2.text(1440, 400, 'EW EFT operator:', fontsize=15)
ax2.text(1590, 300, r'$\frac{\chi \overline{\chi} \ V^{i,\mu \nu}V^i_{\mu \nu}}{\Lambda^3}$', fontsize=15)
ax2.text(1440, 200, 'Dirac Fermion DM', fontsize=15)
ax2.text(1440, 100, r'$R_{miss}=\frac{\sigma_{fid}(p_T^{miss}+jets)}{\sigma_{fid}(l^+ l^- + jets)}$', fontsize=15)
ax2.set_title('Exclusion contours at 95% CL for Dirac-fermion Dark Matter', fontsize=16)
# plt.savefig(folder+'/fig6(top)_replication.pdf')


#  ====================================================================================================
#      G: PLOTS IN PAGE 16 OF DM PAPER with CHI-2
#  ====================================================================================================

err_MG=np.sqrt(np.diag(cov_stat_SM))

#Plots:
fig_a_data=meas[0:7]; fig_b_data=meas[7:13]; fig_c_data=meas[13:18]; fig_d_data=meas[18:24];
fig_a_SM=SM[0:7];     fig_b_SM=SM[7:13];     fig_c_SM=SM[13:18];     fig_d_SM=SM[18:24];    
SM_stat=[0.04, 0.05, 0.03, 0.03, 0.04, 0.07, 0.16]
meas_stat= [(0.13,0.1,0.14,0.13,0.28,0.64,1.4), [0.13, 0.21, 0.27, 0.27, 0.54, 1.52, 3.64]]


# Figure a:
fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
ax.set_title('Figure (a): $R^{miss}$ vs $p_T^{miss}$ for the $\geq 1$ jet region',fontsize=16)     
ax.set_yscale('linear')
ax.set_xlim(200, 1400)
ax.set_ylim(0,12)     
              
pt_miss_bin_centres=[225,275,325,425,600,850,1200]  
bin_edgepoints=[250,300,350,500,700,1000,1400]  
ax.errorbar(pt_miss_bin_centres, fig_a_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax.bar(pt_miss_bin_centres, Rmiss_SM[0:7],fill=False, width=[50,50,50,150,200,300,400], yerr=err_MG[0:7], ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction')
ax.bar(pt_miss_bin_centres, fig_a_SM,fill=False, width=[50,50,50,150,200,300,400], yerr=SM_stat, ecolor='red', edgecolor='red', capsize=4, label='HepData SM prediction')

ax_two=ax.twinx()

ax_two.bar(pt_miss_bin_centres, chi2_MG[0:7], fill=True, width=[50,50,50,150,200,300,400], facecolor='blue', alpha=0.1, label='$\chi^2$ for MG5 (LO) SM')
ax_two.bar(pt_miss_bin_centres, chi2_HEP[0:7], fill=True, width=[50,50,50,150,200,300,400], facecolor='red', alpha=0.1, label='$\chi^2$ for HepData SM')
#ax_two.plot(pt_miss_bin_centres,chi2_MG[0:7], 'x--b', linewidth=1, markersize=10, label='$\chi^2$ for MG5 (LO) SM')              
#ax_two.plot(pt_miss_bin_centres,chi2_HEP[0:7],'x--r', linewidth=1, markersize=10, label='$\chi^2$ for HepData) SM')

ax.set_ylabel('$R^{miss}$', fontsize=15)
ax_two.set_ylabel('$\chi^2$', fontsize=15)
ax.set_xlabel('$p_T^{miss}$ ',fontsize=15)
ax.legend(loc="upper left", fontsize=9)
ax_two.legend(loc="upper right", fontsize=9)
#ax_two.set_ylim(0,12)     
#plt.savefig(folder+'/fig4a_replication.pdf')

# Figure b
SM_stat=[0.08, 0.09, 0.05, 0.05, 0.07, 0.11]
meas_stat= [(0.27,0.21,0.28,0.22,0.42, 0.94), [0.27, 0.42, 0.54, 0.43, 0.79, 2.50]]
fig, ax3 = plt.subplots(1,figsize=(9,6),sharex=True)
ax3.set_title('Figure (b): $R^{miss}$ vs $p_T^{miss}$ for the $VBF$ region',fontsize=16)     
ax3.set_yscale('linear')
ax3.set_xlim(200, 1400)
ax3.set_ylim(0,12)     
              
pt_miss_bin_centres_2=[225,275,325,425,600,1050]
bin_edgepoints_2=[250,300,350,500,700,1400]    
ax3.errorbar(pt_miss_bin_centres_2, fig_b_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax3.bar(pt_miss_bin_centres_2, Rmiss_SM[7:13], fill=False, width=[50,50,50,150,200,700], yerr=err_MG[7:13], ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction')
ax3.bar(pt_miss_bin_centres_2, fig_b_SM,fill=False, width=[50,50,50,150,200,700], yerr=SM_stat, ecolor='red', edgecolor='red', capsize=4, label='HepData SM prediction')

ax3_two=ax3.twinx()
ax3_two.bar(pt_miss_bin_centres_2, chi2_MG[7:13], fill=True, width=[50,50,50,150,200,700], facecolor='blue', alpha=0.1, label='$\chi^2$ for MG5 (LO) SM')
ax3_two.bar(pt_miss_bin_centres_2, chi2_HEP[7:13], fill=True, width=[50,50,50,150,200,700], facecolor='red', alpha=0.1, label='$\chi^2$ for HepData SM')

#ax3_two.plot(pt_miss_bin_centres_2,chi2_MG[7:13], 'x--b', linewidth=1, markersize=10, label='$\chi^2$ for MG5 (LO) SM')              
#ax3_two.plot(pt_miss_bin_centres_2,chi2_HEP[7:13],'x--r', linewidth=1, markersize=10, label='$\chi^2$ for HepData SM')
ax3_two.set_ylabel(r'$\chi^2$', fontsize=15)

ax3.set_ylabel('$R^{miss}$', fontsize=15)
ax3.set_xlabel('$p_T^{miss}$ ',fontsize=15)
ax3.legend(loc="upper left", fontsize=9)
ax3_two.legend(loc="upper right", fontsize=9)
#ax3_two.set_ylim(0,12)
#plt.savefig(folder+'/fig4b_replication.pdf')

# Figure c:
SM_stat=[0.06, 0.09, 0.11, 0.15, 0.43]
meas_stat= [(0.27,0.19,0.24,0.41,1.18), [0.27, 0.36, 0.44, 0.78, 2.60]]
fig, ax4 = plt.subplots(1,figsize=(9,6),sharex=True)
ax4.set_title('Figure (c): $R^{miss}$ vs $m_{jj}$ for the $VBF$ region',fontsize=16)     
ax4.set_yscale('linear')
ax4.set_xlim(200, 4000)
ax4.set_ylim(0,12)     
              
pt_miss_bin_centres_3=[300,500,800,1500,3000] 
bin_edgepoints_3=[400,600,1000,2000,4000]      
ax4.errorbar(pt_miss_bin_centres_3, fig_c_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax4.bar(pt_miss_bin_centres_3, Rmiss_SM[13:18],fill=False, width=[200,200,400,1000,2000], yerr=err_MG[13:18], ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction')
ax4.bar(pt_miss_bin_centres_3, fig_c_SM,fill=False, width=[200,200,400,1000,2000], yerr=SM_stat, ecolor='red', edgecolor='red', capsize=4, label='HepData SM prediction')

#Chi2:
ax4_two=ax4.twinx()
ax4_two.bar(pt_miss_bin_centres_3, chi2_MG[13:18], fill=True, width=[200,200,400,1000,2000], facecolor='blue', alpha=0.1, label='$\chi^2$ for MG5 (LO) SM')
ax4_two.bar(pt_miss_bin_centres_3, chi2_HEP[13:18], fill=True, width=[200,200,400,1000,2000], facecolor='red', alpha=0.1, label='$\chi^2$ for HepData SM')

#ax4_two.plot(pt_miss_bin_centres_3,chi2_MG[13:18], 'x--b',linewidth=1, markersize=10, label='$\chi^2$ for MG5 (LO) SM')              
#ax4_two.plot(pt_miss_bin_centres_3,chi2_HEP[13:18],'x--r',linewidth=1, markersize=10, label='$\chi^2$ for HepData SM')

ax4_two.set_ylabel(r'$\chi^2$', fontsize=15)
ax4_two.legend(loc="upper right", fontsize=9)

ax4.set_ylabel('$R^{miss}$', fontsize=15)
ax4.set_xlabel('$m_{jj}$ ',fontsize=15)
ax4.legend(loc="upper left", fontsize=8)
#plt.savefig(folder+'/fig4c_replication.pdf')

#Figure d:
SM_stat=[0.12, 0.12, 0.10, 0.09, 0.10, 0.14]
meas_stat= [(0.46,0.25,0.22,0.19,0.25, 0.33), [0.46, 0.48, 0.45, 0.36, 0.47, 0.59]]
fig, ax5 = plt.subplots(1,figsize=(9,6),sharex=True)
ax5.set_title('Figure (d): $R^{miss}$ vs $\Delta \phi_{jj}$ for the $VBF$ region',fontsize=16)     
ax5.set_yscale('linear')
ax5.set_xlim(0, np.pi)
ax5.set_ylim(0,12)     

pt_miss_bin_centres_4=[]
bin_edgepoints_4=[]   
for n in range(6):
    pt_miss_bin_centres_4.append(np.pi/12+n*np.pi/6)
    bin_edgepoints_4.append(np.pi/6+n*np.pi/6)
ax5.errorbar(pt_miss_bin_centres_4, fig_d_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax5.bar(pt_miss_bin_centres_4, Rmiss_SM[18:24],fill=False, width=np.pi/6, yerr=err_MG[18:24], ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction')
ax5.bar(pt_miss_bin_centres_4, fig_d_SM,fill=False, width=np.pi/6, yerr=SM_stat, ecolor='red', edgecolor='red', capsize=4, label='HepData SM prediction')

ax5_two=ax5.twinx()
ax5_two.bar(pt_miss_bin_centres_4, chi2_MG[18:24], fill=True, width=np.pi/6, facecolor='blue', alpha=0.1, label='$\chi^2$ for MG5 (LO) SM')
ax5_two.bar(pt_miss_bin_centres_4, chi2_HEP[18:24], fill=True, width=np.pi/6, facecolor='red', alpha=0.1, label='$\chi^2$ for HepData SM')
#ax5_two.plot(pt_miss_bin_centres_4,chi2_MG[18:24], 'x--b', linewidth=1, markersize=10, label='$\chi^2$ for MG5 (LO) SM')              
#ax5_two.plot(pt_miss_bin_centres_4,chi2_HEP[18:24],'x--r', linewidth=1, markersize=10, label='$\chi^2$ for HepData SM')


ax5_two.set_ylabel(r'$\chi^2$', fontsize=15)
ax5_two.legend(loc="upper right", fontsize=9)

ax5.set_ylabel('$R^{miss}$', fontsize=15)
ax5.set_xlabel('$\Delta \phi_{jj}$',fontsize=15)
ax5.legend(loc="upper left", fontsize=8)
#plt.savefig(folder+'/fig4d_replication.pdf')


#  ====================================================================================================
#      G: DELTA CHI2 PLOTS
#  ====================================================================================================
Delchi2_OUT = get_deltachi2_perbin(850)

Delchi2_HEP = Delchi2_OUT[0]  # At lambda 800
Delchi2_MG  = Delchi2_OUT[1]  
BSM         = Delchi2_OUT[2]  
BSM_MG      = Delchi2_OUT[3]
err_BSM     = Delchi2_OUT[4]
err_BSM_MG  = Delchi2_OUT[4]

err_SM=np.sqrt(np.diag(SM_exp_cov))


print(Delchi2_HEP)

#Plots:
fig_a_data=meas[0:7]; fig_b_data=meas[7:13]; fig_c_data=meas[13:18]; fig_d_data=meas[18:24];
fig_a_SM=SM[0:7];     fig_b_SM=SM[7:13];     fig_c_SM=SM[13:18];     fig_d_SM=SM[18:24];    
SM_stat=[0.04, 0.05, 0.03, 0.03, 0.04, 0.07, 0.16]
meas_stat= [(0.13,0.1,0.14,0.13,0.28,0.64,1.4), [0.13, 0.21, 0.27, 0.27, 0.54, 1.52, 3.64]]

# Figure a:
fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
ax.set_title('Figure (a): $R^{miss}$ vs $p_T^{miss}$ for the $\geq 1$ jet region',fontsize=16)     
ax.set_yscale('linear')
ax.set_xlim(200, 1400)
ax.set_ylim(0,12)     
              
pt_miss_bin_centres=[225,275,325,425,600,850,1200]  
bin_edgepoints=[250,300,350,500,700,1000,1400]  
ax.errorbar(pt_miss_bin_centres, fig_a_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax.bar(pt_miss_bin_centres, SM[0:7],fill=False, width=[50,50,50,150,200,300,400], yerr=err_SM[0:7], ecolor='blue', edgecolor='blue', capsize=4, label='HepData SM prediction')
#ax.bar(pt_miss_bin_centres, fig_a_SM,fill=False, width=[50,50,50,150,200,300,400], yerr=SM_stat, ecolor='red', edgecolor='red', capsize=4, label='HepData SM prediction')
ax.bar(pt_miss_bin_centres, BSM[0:7],fill=False, width=[50,50,50,150,200,300,400], yerr=err_BSM[0:7], ecolor='green', edgecolor='green', capsize=4, label='MG5(LO) SM+DM prediction')

ax_two=ax.twinx()

#ax_two.bar(pt_miss_bin_centres, Delchi2_MG[0:7], fill=True, width=[50,50,50,150,200,300,400], facecolor='blue', alpha=0.1, label='$\Delta \chi^2$ for MG5 (LO)')
ax_two.bar(pt_miss_bin_centres, Delchi2_HEP[0:7], fill=True, width=[50,50,50,150,200,300,400], facecolor='red', alpha=0.1, label='$\Delta \chi^2$ for HepData SM')

ax.set_ylabel('$R^{miss}$', fontsize=15)
ax_two.set_ylabel(r'$\mid \chi^2_{BSM}-\chi^2_{SM} \mid$', fontsize=15)
ax.set_xlabel('$p_T^{miss}$ ',fontsize=15)
ax.legend(loc="upper left", fontsize=9)
ax_two.legend(loc="upper right", fontsize=9)
# ax_two.set_ylim(0,12)     

#plt.savefig(folder+'/fig4a_delchi2.pdf')

# Figure B:
SM_stat=[0.08, 0.09, 0.05, 0.05, 0.07, 0.11]
meas_stat= [(0.27,0.21,0.28,0.22,0.42, 0.94), [0.27, 0.42, 0.54, 0.43, 0.79, 2.50]]
fig, ax3 = plt.subplots(1,figsize=(9,6),sharex=True)
ax3.set_title('Figure (b): $R^{miss}$ vs $p_T^{miss}$ for the $VBF$ region',fontsize=16)     
ax3.set_yscale('linear')
ax3.set_xlim(200, 1400)
ax3.set_ylim(0,12)     
              
pt_miss_bin_centres_2=[225,275,325,425,600,1050]
bin_edgepoints_2=[250,300,350,500,700,1400]    
ax3.errorbar(pt_miss_bin_centres_2, fig_b_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax3.bar(pt_miss_bin_centres_2, SM[7:13], fill=False, width=[50,50,50,150,200,700], yerr=err_SM[7:13], ecolor='blue', edgecolor='blue', capsize=4, label='HepData SM prediction')
#ax3.bar(pt_miss_bin_centres_2, fig_b_SM,fill=False, width=[50,50,50,150,200,700], yerr=SM_stat, ecolor='red', edgecolor='red', capsize=4, label='HepData SM prediction')
ax3.bar(pt_miss_bin_centres_2, BSM[7:13],fill=False, width=[50,50,50,150,200,700], yerr=err_BSM[7:13], ecolor='green', edgecolor='green', capsize=4, label='MG5(LO) SM+DM prediction')

ax3_two=ax3.twinx()
#ax3_two.bar(pt_miss_bin_centres_2, Delchi2_MG[7:13], fill=True, width=[50,50,50,150,200,700], facecolor='blue', alpha=0.1, label='$\Delta \chi^2$ for MG5 (LO)')
ax3_two.bar(pt_miss_bin_centres_2, Delchi2_HEP[7:13], fill=True, width=[50,50,50,150,200,700], facecolor='red', alpha=0.1, label='$\Delta \chi^2$ for HepData SM')

ax3_two.set_ylabel(r'$\chi^2$', fontsize=15)

ax3.set_ylabel('$R^{miss}$', fontsize=15)
ax3_two.set_ylabel(r'$\mid \chi^2_{BSM}-\chi^2_{SM} \mid$', fontsize=15)
ax3.set_xlabel('$p_T^{miss}$ ',fontsize=15)
ax3.legend(loc="upper left", fontsize=9)
ax3_two.legend(loc="upper right", fontsize=9)     
#plt.savefig(folder+'/fig4b_delchi2.pdf')



# Figure c:
SM_stat=[0.06, 0.09, 0.11, 0.15, 0.43]
meas_stat= [(0.27,0.19,0.24,0.41,1.18), [0.27, 0.36, 0.44, 0.78, 2.60]]
fig, ax4 = plt.subplots(1,figsize=(9,6),sharex=True)
ax4.set_title('Figure (c): $R^{miss}$ vs $m_{jj}$ for the $VBF$ region',fontsize=16)     
ax4.set_yscale('linear')
ax4.set_xlim(200, 4000)
ax4.set_ylim(0,12)     
              
pt_miss_bin_centres_3=[300,500,800,1500,3000] 
bin_edgepoints_3=[400,600,1000,2000,4000]      
ax4.errorbar(pt_miss_bin_centres_3, fig_c_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax4.bar(pt_miss_bin_centres_3, SM[13:18],fill=False, width=[200,200,400,1000,2000], yerr=err_SM[13:18], ecolor='blue', edgecolor='blue', capsize=4, label='HepData SM prediction')
# ax4.bar(pt_miss_bin_centres_3, fig_c_SM,fill=False, width=[200,200,400,1000,2000], yerr=SM_stat, ecolor='red', edgecolor='red', capsize=4, label='HepData SM prediction')
ax4.bar(pt_miss_bin_centres_3, BSM[13:18],fill=False, width=[200,200,400,1000,2000], yerr=err_BSM[13:18], ecolor='green', edgecolor='green', capsize=4, label='MG5(LO) SM+DM prediction')

ax4_two=ax4.twinx()
#ax4_two.bar(pt_miss_bin_centres_3, Delchi2_MG[13:18], fill=True, width=[200,200,400,1000,2000], facecolor='blue', alpha=0.1, label='$\Delta \chi^2$ for MG5 (LO)')
ax4_two.bar(pt_miss_bin_centres_3, Delchi2_HEP[13:18], fill=True, width=[200,200,400,1000,2000], facecolor='red', alpha=0.1, label='$\chi^2$ for HepData SM')

ax4_two.set_ylabel(r'$\mid \chi^2_{BSM}-\chi^2_{SM} \mid$', fontsize=15)
ax4_two.legend(loc="upper right", fontsize=9)

ax4.set_ylabel('$R^{miss}$', fontsize=15)
ax4.set_xlabel('$m_{jj}$ ',fontsize=15)
ax4.legend(loc="upper left", fontsize=8)
# plt.savefig(folder+'/fig4c_delchi2.pdf')


#Figure d:
SM_stat=[0.12, 0.12, 0.10, 0.09, 0.10, 0.14]
meas_stat= [(0.46,0.25,0.22,0.19,0.25, 0.33), [0.46, 0.48, 0.45, 0.36, 0.47, 0.59]]
fig, ax5 = plt.subplots(1,figsize=(9,6),sharex=True)
ax5.set_title('Figure (d): $R^{miss}$ vs $\Delta \phi_{jj}$ for the $VBF$ region',fontsize=16)     
ax5.set_yscale('linear')
ax5.set_xlim(0, np.pi)
ax5.set_ylim(0,12)     

pt_miss_bin_centres_4=[]
bin_edgepoints_4=[]   
for n in range(6):
    pt_miss_bin_centres_4.append(np.pi/12+n*np.pi/6)
    bin_edgepoints_4.append(np.pi/6+n*np.pi/6)
ax5.errorbar(pt_miss_bin_centres_4, fig_d_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax5.bar(pt_miss_bin_centres_4, Rmiss_SM[18:24],fill=False, width=np.pi/6, yerr=err_MG[18:24], ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction')
# ax5.bar(pt_miss_bin_centres_4, fig_d_SM,fill=False, width=np.pi/6, yerr=SM_stat, ecolor='red', edgecolor='red', capsize=4, label='HepData SM prediction')
ax5.bar(pt_miss_bin_centres_4, BSM[18:24],fill=False, width=np.pi/6, yerr=err_BSM[18:24], ecolor='green', edgecolor='green', capsize=4, label='MG5(LO) SM+DM prediction')

ax5_two=ax5.twinx()
ax5_two.bar(pt_miss_bin_centres_4, Delchi2_MG[18:24], fill=True, width=np.pi/6, facecolor='blue', alpha=0.1, label='$\Delta \chi^2$ for MG5 (LO)')
#ax5_two.bar(pt_miss_bin_centres_4, Delchi2_HEP[18:24], fill=True, width=np.pi/6, facecolor='red', alpha=0.1, label='$\chi^2$ for HepData SM')

ax5_two.set_ylabel(r'$\mid \chi^2_{BSM}-\chi^2_{SM} \mid$', fontsize=15)
ax5_two.legend(loc="upper right", fontsize=9)

ax5.set_ylabel('$R^{miss}$', fontsize=15)
ax5.set_xlabel('$\Delta \phi_{jj}$',fontsize=15)
ax5.legend(loc="upper left", fontsize=8)
# plt.savefig(folder+'/fig4d_delchi2.pdf')



