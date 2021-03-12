# Calculating Fisher information from d^2 ln L / d lda^2:
import sys
import pickle
import numpy             as np
import scipy.stats       as stats
import matplotlib.pyplot as plt

save_to='/Users/Genis/mphys/Rivet/DM/HNoE/plots/plots_dec_29'; 
# =============================================================================
#                   IMPORT FROM PICKLE FILES:
# =============================================================================
# ATLAS Data:
inputs = pickle.load(open("paper_values_and_DMEFT_D7a_m10_l400_prediction.pickle", "rb"))

meas         = inputs["meas_values"]              #  This is the measurement (central values)
meas_cov     = inputs["meas_cov"]                 #  This is the measurement (covariance)
meas_cov_inv = np.linalg.inv(meas_cov)            #  This is the inverse covariance
n_dof        = len(meas)                          #  This is the number of bins (=24)
SM           = inputs["SM_model_values"]          #  This is the SM expectation (central values)
SM_exp_cov   = inputs["SM_exp_cov"]               #  This is the expected experimental covariance for ATLAS
SM_thry_cov  = inputs["SM_model_cov"]             #  This is the covariance on the central values due to theory modelling
SM_thry_cov  = np.zeros(shape=(n_dof, n_dof))               # Uncomment this line if you want to ignore theory uncertainty on SM
SM_total_cov = SM_exp_cov + SM_thry_cov           #  This is the sum of the theory covariance with the expected experimental covariance
SM_exp_cov_inv   = np.linalg.inv(SM_exp_cov)      
SM_total_cov_inv = np.linalg.inv(SM_total_cov)

# Available pickle files are: MG_Data_m100, MG_Data_m300, MG_Data_m500,
#                             MG_Data_m700, MG_Data_m900.
    
with open('MG_Data_m100.pickle', 'rb') as f:
    inputs = pickle.load(f)

mchi        = inputs[0]    # mchi used
lda         = inputs[1]    # Lambda used 
Rmiss_SM    = inputs[2]    # MG5(LO) Rmiss for SM prediction
Rmiss_DM    = inputs[3]    # MG5(LO) Rmiss for DM only prediction.
cov_stat_SM = inputs[4]    # Statistical Covariance Matrix for SM prediction (MG)
cov_stat_DM = inputs[5]    # Statistical Covariance Matrix for DM prediction (MG)

BSM1       = [lda**6*DM400 for DM400 in Rmiss_DM] # DM at lambda=1: Used to calc. F.I.

# SM PREDICTION & EXPERIMENTAL ERROR:
x          = meas
sigma_x    = np.sqrt(np.diag(meas_cov))
# x          = Rmiss_SM                      # SM Hypothesis
# tot_err_x  = SM_exp_cov+cov_stat_SM        # SM error
# sigma_x    = np.sqrt(np.diag(tot_err_x))   # Expected Experimental Covariance.

# BSM PREDICTION &  ERRORS:
p400       = [x+y for x,y in zip(SM, Rmiss_DM)]   # SM+DM prediction at l=400
p600       = [x+y*(400/600)**6 for x,y in zip(SM, Rmiss_DM)] # SM+DM prediction at l=800
p800       = [x+y*(400/800)**6 for x,y in zip(SM, Rmiss_DM)] # SM+DM prediction at l=800

tot_cov400 = [x+y for x,y in zip(SM_exp_cov, cov_stat_DM)]  # SM+DM MG5 erros. 
tot_cov600 = [x+y*(400/600)**6*(400/600)**6 for x,y in zip(SM_exp_cov, cov_stat_DM)]  # SM+DM MG5 errors. 
tot_cov800 = [x+y*(400/800)**6*(400/800)**6 for x,y in zip(SM_exp_cov, cov_stat_DM)]  # SM+DM MG5 errors. 

sigma_p400 = np.sqrt(np.diag(tot_cov400))  # Errors for SM+DM at l=400
sigma_p600 = np.sqrt(np.diag(tot_cov600))  # Errors for SM+DM at l=800
sigma_p800 = np.sqrt(np.diag(tot_cov800))  # Errors for SM+DM at l=800

# DM ONLY PREDICTION &  ERRORS:
DM400       = Rmiss_DM                             # DM online at l=400
DM600       = [y*(400/600)**6 for y in Rmiss_DM]   # DM online at l=600
DM800       = [y*(400/800)**6 for y in Rmiss_DM]   # DM online at l=800

DM_cov400   = cov_stat_DM
DM_cov600   = [y*(400/600)**6*(400/600)**6 for y in cov_stat_DM]  # DM MG5 errors at l600. 
DM_cov800   = [y*(400/800)**6*(400/800)**6 for y in cov_stat_DM]  # DM MG5 errors at l800. 

sigma_DM400 = np.sqrt(np.diag(DM_cov400))
sigma_DM600 = np.sqrt(np.diag(DM_cov600))
sigma_DM800 = np.sqrt(np.diag(DM_cov800))

# -----------------------------------------------------------------------------    
def throw_toys(mean, std_dev, ntoys, nbins=30, plot=True):
    toys=[]; weights=[];
    for i in range(ntoys):
        ith_toy=np.random.normal(loc=mean, scale=std_dev)
        toys.append(ith_toy)
    mean_toys=np.sum(toys)/ntoys
    
    if plot==True:
        figt, axt = plt.subplots(1,figsize=(9,6),sharex=True)
        
        # Create temporary histogram to calculate normalisation:
        h_temp=plt.hist(toys, bins=nbins, histtype='bar', label='Toys')        
        bin_width=h_temp[1][1]-h_temp[1][0]
        areas=bin_width*h_temp[0]
        # Normalise to area = 1
        norm_factor=1.0/np.sum(areas); 
        weights[:]=[norm_factor for i in range(ntoys)]
        
        plt.close('all')
        del figt, axt, h_temp
        
        fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
        
        ax.set_title('Toys arround measured data',fontsize=16)     
        ax.set_yscale('linear') 
        
        x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 200)
        ax.plot(x, stats.norm.pdf(x, mean, std_dev), label='Expected Gaussian')
        ax.hist(toys, bins=nbins, weights=weights, histtype='bar', label='Toys')                 
        ax.text(0.03, 0.93, 'ntoys = '+str(ntoys), fontsize=15, transform=ax.transAxes)
        ax.text(0.03, 0.87, 'Toys mean = '+str(np.round(mean_toys, decimals=2)), fontsize=15, transform=ax.transAxes)
        ax.set_ylabel('Probability', fontsize=15)
        ax.set_xlabel('Toy Frequency',fontsize=15)
        ax.legend(loc="upper right", fontsize=13)
        plt.savefig(save_to+'/Test_Toys.pdf')
    plt.show()
    return(toys)


def dlogLsquared(x, lda, p, BSM1, sigma_x, sigma_p):
    # return d^2 ln L / d lda^2 assuming Gaussian L.  
    return(((x-p)/(sigma_x**2+sigma_p**2))*(42*BSM1/lda**8)-(36*BSM1*BSM1/(lda**14*(sigma_x**2+sigma_p**2))))
    # Throw toys to find average. 

def get_info(x, lda, p, BSM1, sigma_x, sigma_p, ntoys=5000, plot_dist=True):
    toys=throw_toys(mean=x, std_dev=sigma_x, ntoys=ntoys, nbins=50, plot=False)

    All_dlogLsquared=[]
    for i in range(ntoys):
        x=toys[i];
        dlogLsquared_pertoy = dlogLsquared(x, lda, p, BSM1, sigma_x, sigma_p)
        All_dlogLsquared.append(dlogLsquared_pertoy)
    mean=np.sum(All_dlogLsquared)/ntoys
    
    if plot_dist==True:  
        fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
        
        ax.set_title('Toys arround measured data',fontsize=16)     
        ax.set_yscale('linear') 
        
        ax.hist(All_dlogLsquared, bins=50, histtype='bar', label='Toys')                 
        ax.text(0.03, 0.93, 'ntoys = '+str(ntoys), fontsize=15, transform=ax.transAxes)
        ax.text(0.03, 0.87, 'Toys mean = '+str(np.round(mean, decimals=5)), fontsize=15, transform=ax.transAxes)
        ax.set_ylabel('Toy Frequency', fontsize=15)
        ax.set_xlabel(r'$\frac{d^2 ln\ L}{d \Lambda_{EFT}^2}$',fontsize=15)
        ax.legend(loc="upper right", fontsize=13)
        plt.show()
    
    
    #print('Information:', -1.0*mean)
    return (-1.0*mean)   


throw_toys(7.0, 1.0, 10000, nbins=50)
    

info_all_bins=[]
for i in range(len(x)):
    info_bin_i=get_info(x[i],800, p800[i], BSM1[i], sigma_x[i], sigma_p800[i], plot_dist=False)
    info_all_bins.append(info_bin_i)
#print('Information:', info_all_bins)

#------------------------------------------------------------------------------

err_MG=np.sqrt(np.diag(cov_stat_SM))

#Plots:
fig_a_data=meas[0:7]; fig_b_data=meas[7:13]; fig_c_data=meas[13:18]; fig_d_data=meas[18:24];
fig_a_SM=SM[0:7];     fig_b_SM=SM[7:13];     fig_c_SM=SM[13:18];     fig_d_SM=SM[18:24];    
SM_stat=[0.04, 0.05, 0.03, 0.03, 0.04, 0.07, 0.16]
meas_stat= [(0.13,0.1,0.14,0.13,0.28,0.64,1.4), [0.13, 0.21, 0.27, 0.27, 0.54, 1.52, 3.64]]

print('Info plot a: ', np.round(np.sum(info_all_bins[0:7])*1000, 3), 'E-3')
print('Info plot b: ', np.round(np.sum(info_all_bins[7:13])*1000, 3), 'E-3')
print('Info plot c: ', np.round(np.sum(info_all_bins[13:18])*1000, 3), 'E-3')
print('Info plot d: ', np.round(np.sum(info_all_bins[18:24])*1000, 3), 'E-3')

# Figure a:
all_tog = SM[0:7]+DM600[0:7];
all_tog_err = (np.sqrt(x**2+y**2) for x,y in zip(SM_stat, sigma_DM600[0:7]))

fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
ax.set_title('Figure (a): $R^{miss}$ vs $p_T^{miss}$ for the $\geq 1$ jet region',fontsize=16)     
ax.set_yscale('linear')
ax.set_xlim(200, 1400)
ax.set_ylim(0,20)  

ax.set_ylabel('$R^{miss}$', fontsize=15)
ax.set_xlabel('$p_T^{miss}$ ',fontsize=15)
              
bin_centres=[225,275,325,425,600,850,1200]  
bin_edgepoints=[250,300,350,500,700,1000,1400]  
ax.errorbar(bin_centres, fig_a_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax.bar(bin_centres, SM[0:7],fill=False, width=[50,50,50,150,200,300,400], yerr=SM_stat, ecolor='blue', edgecolor='blue', capsize=4, label='HepData SM (with exp. experimental errors)')
ax.bar(bin_centres, DM600[0:7], fill=False, width=[50,50,50,150,200,300,400], linestyle='--', yerr=sigma_DM600[0:7], ecolor='red', edgecolor='red', capsize=4, label='MG5(LO) DM at $m_{\chi}$=100, $\Lambda_{EFT}$=800')
ax.bar(bin_centres, all_tog, fill=False, width=[50,50,50,150,200,300,400], linestyle='--', yerr=all_tog_err, ecolor='green', edgecolor='green', capsize=4, label='SM+DM prediction')

ax_two=ax.twinx()
ax_two.bar(bin_centres, info_all_bins[0:7], fill=True, alpha=0.2, width=[50,50,50,150,200,300,400], facecolor='orange', label='Fisher Information')

ax_two.set_ylabel('Fisher Information')
ax_two.set_ylim(0)

ax.legend(loc="upper left", fontsize=8)   
ax_two.legend(loc="upper right", fontsize=8)
plt.savefig(save_to+'/Measured_info_a'+str(mchi)+'l800.pdf')

# Figure b
SM_stat=[0.08, 0.09, 0.05, 0.05, 0.07, 0.11]
all_tog = SM[7:13]+DM600[7:13];
all_tog_err = (np.sqrt(x**2+y**2) for x,y in zip(SM_stat, sigma_DM600[7:13]))
meas_stat= [(0.27,0.21,0.28,0.22,0.42, 0.94), [0.27, 0.42, 0.54, 0.43, 0.79, 2.50]]
fig, ax3 = plt.subplots(1,figsize=(9,6),sharex=True)
ax3.set_title('Figure (b): $R^{miss}$ vs $p_T^{miss}$ for the $VBF$ region',fontsize=16)     
ax3.set_yscale('linear')
ax3.set_xlim(200, 1400)
ax3.set_ylim(0,18)     
              
bin_centres_2=[225,275,325,425,600,1050]
bin_edgepoints_2=[250,300,350,500,700,1400]    
ax3.errorbar(bin_centres_2, fig_b_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax3.bar(bin_centres_2, SM[7:13], fill=False, width=[50,50,50,150,200,700], yerr=SM_stat, ecolor='blue', edgecolor='blue', capsize=4, label='HepData SM (with exp. experimental errors)')
ax3.bar(bin_centres_2, DM600[7:13], fill=False, width=[50,50,50,150,200,700], linestyle='--', yerr=sigma_DM600[7:13], ecolor='red', edgecolor='red', capsize=4, label='MG5(LO) DM at $m_{\chi}$=100, $\Lambda_{EFT}$=800')
ax3.bar(bin_centres_2, all_tog, fill=False, width=[50,50,50,150,200,700], linestyle='--', yerr=all_tog_err, ecolor='green', edgecolor='green', capsize=4, label='SM+DM prediction')

ax3_two=ax3.twinx()
ax3_two.bar(bin_centres_2, info_all_bins[7:13], fill=True, alpha=0.2, width=[50,50,50,150,200,700], facecolor='orange', label='Fisher Information')
ax3_two.set_ylabel('Fisher Information')
ax3_two.set_ylim(0)


ax3.set_ylabel('$R^{miss}$', fontsize=15)
ax3.set_xlabel('$p_T^{miss}$ ',fontsize=15)
ax3.legend(loc="upper left", fontsize=8)
ax3_two.legend(loc="upper right", fontsize=8)
plt.savefig(save_to+'/Measured_info_b'+str(mchi)+'l800.pdf')



# Figure c:
SM_stat=[0.06, 0.09, 0.11, 0.15, 0.43]
all_tog = SM[13:18]+DM600[13:18];
all_tog_err = (np.sqrt(x**2+y**2) for x,y in zip(SM_stat, sigma_DM600[13:18]))
meas_stat= [(0.27,0.19,0.24,0.41,1.18), [0.27, 0.36, 0.44, 0.78, 2.60]]
fig, ax4 = plt.subplots(1,figsize=(9,6),sharex=True)
ax4.set_title('Figure (c): $R^{miss}$ vs $m_{jj}$ for the $VBF$ region',fontsize=16)     
ax4.set_yscale('linear')
ax4.set_xlim(200, 4000)
ax4.set_ylim(0,20)     
              
bin_centres_3=[300,500,800,1500,3000] 
bin_edgepoints_3=[400,600,1000,2000,4000]      
ax4.errorbar(bin_centres_3, fig_c_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax4.bar(bin_centres_3, SM[13:18],fill=False, width=[200,200,400,1000,2000], yerr=SM_stat, ecolor='blue', edgecolor='blue', capsize=4, label='HepData SM (with exp. experimental errors)')
ax4.bar(bin_centres_3, DM600[13:18], fill=False, width=[200,200,400,1000,2000], linestyle='--', edgecolor='red', yerr=sigma_DM600[13:18], ecolor='red', capsize=4, label='MG5(LO) DM at $m_{\chi}$=100, $\Lambda_{EFT}$=800')
ax4.bar(bin_centres_3, all_tog, fill=False, width=[200,200,400,1000,2000], linestyle='--', yerr=all_tog_err, ecolor='green', edgecolor='green', capsize=4, label='SM+DM prediction')

ax4_two=ax4.twinx()
ax4_two.bar(bin_centres_3, info_all_bins[13:18], fill=True, alpha=0.2, width=[200,200,400,1000,2000], facecolor='orange', label='Fisher Information')
ax4_two.set_ylabel('Fisher Information')
ax4_two.set_ylim(0)

ax4.set_ylabel('$R^{miss}$', fontsize=15)
ax4.set_xlabel('$m_{jj}$ ',fontsize=15)
ax4.legend(loc="upper left", fontsize=8)
ax4_two.legend(loc="upper right", fontsize=8)
plt.savefig(save_to+'/Measured_info_c'+str(mchi)+'l800.pdf')

#Figure d:
SM_stat=[0.12, 0.12, 0.10, 0.09, 0.10, 0.14]
all_tog = SM[18:24]+DM600[18:24];
all_tog_err = (np.sqrt(x**2+y**2) for x,y in zip(SM_stat, sigma_DM600[18:24]))

meas_stat= [(0.46,0.25,0.22,0.19,0.25, 0.33), [0.46, 0.48, 0.45, 0.36, 0.47, 0.59]]
fig, ax5 = plt.subplots(1,figsize=(9,6),sharex=True)
ax5.set_title('Figure (d): $R^{miss}$ vs $\Delta \phi_{jj}$ for the $VBF$ region',fontsize=16)     
ax5.set_yscale('linear')
ax5.set_xlim(0, np.pi)
ax5.set_ylim(0,12)  

bin_centres_4=[]
bin_edgepoints_4=[]   
for n in range(6):
    bin_centres_4.append(np.pi/12+n*np.pi/6)
    bin_edgepoints_4.append(np.pi/6+n*np.pi/6)
ax5.errorbar(bin_centres_4, fig_d_data, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
ax5.bar(bin_centres_4, SM[18:24],fill=False, width=np.pi/6, yerr=SM_stat, ecolor='blue', edgecolor='blue', capsize=4, label='HepData SM (with exp. experimental errors)')
ax5.bar(bin_centres_4, DM600[18:24], fill=False, width=np.pi/6, linestyle='--', yerr=sigma_DM600[18:24], ecolor='red', edgecolor='red', capsize=4, label='MG5(LO) DM at $m_{\chi}$=100, $\Lambda_{EFT}$=800')
ax5.bar(bin_centres_4, all_tog, fill=False, width=np.pi/6, linestyle='--', yerr=all_tog_err, ecolor='green', edgecolor='green', capsize=4, label='SM+DM prediction')

ax5_two=ax5.twinx()

ax5_two.bar(bin_centres_4, info_all_bins[18:24], fill=True, alpha=0.2, width=np.pi/6, facecolor='orange', label='Fisher Information')
ax5_two.set_ylabel('Fisher Information')
ax5_two.set_ylim(0)   

ax5.set_ylabel('$R^{miss}$', fontsize=15)
ax5.set_xlabel('$\Delta \phi_{jj}$',fontsize=15)
ax5.legend(loc="upper left", fontsize=8)
ax5_two.legend(loc="upper right", fontsize=8)
plt.savefig(save_to+'/Measured_info_d'+str(mchi)+'l800.pdf')
