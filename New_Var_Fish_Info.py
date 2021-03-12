# Calculating Fisher information from d^2 ln L / d lda^2:
# NEW MG ONLY DATA:
import pickle
import numpy             as np
import scipy.stats       as stats
import Functions_library as fl
import matplotlib.pyplot as plt

save_to='/Users/Genis/mphys/Rivet/DM/HNoE/plots/plots_dec_12'; 
# =============================================================================
#                   IMPORT FROM PICKLE FILES:
# =============================================================================
# OPEN MG5(LO) Data
with open('MG_Data_jpt1_MONO.pickle', 'rb') as f:
    inputs = pickle.load(f)

mchi         = inputs[0]    # mchi used
lda          = inputs[1]    # Lambda used 
DM_var       = inputs[2]    # MG5(LO) Rmiss for SM prediction
SM_var       = inputs[3]    # MG5(LO) Rmiss for DM only prediction.
DM_var_err   = inputs[4]    # Statistical Covariance Matrix for SM prediction (MG)
SM_var_err   = inputs[5]    # Statistical Covariance Matrix for DM prediction (MG)

BSM1       = [lda**6*DM400 for DM400 in DM_var] # DM at lambda=1: Used to calc. F.I.

# SM PREDICTION & EXPERIMENTAL ERROR:
x          = SM_var                      # SM hypothesis
# Use 10% expected experimental covariance for now...
sigma_x    = [0.1*x for x in SM_var]  # Expected Experimental Covariance.

# BSM PREDICTION & (MADGRAPH) ERRORS:
p400       = [x+y for x,y in zip(SM_var, DM_var)]                  # SM+DM prediction at l=400.
sigma_p400 = [np.sqrt(x+y) for x,y in zip(SM_var_err, DM_var_err)] # Errors for SM+DM at l=400.

p800       = [x+y*(400/800)**6 for x,y in zip(SM_var, DM_var)]         # SM+DM prediction at l=800.
sigma_p800 = [np.sqrt(x+y*(400/800)**6) for x,y in zip(SM_var_err, DM_var_err)] # Errors for SM+DM at l=800.

# DM ONLY PREDICTION &  ERRORS:
DM400       = DM_var 
sigma_DM400 = DM_var_err

DM800       = [y*(400/800)**6 for y in DM_var]       # DM only at l=800
sigma_DM800 = [y*(400/800)**6 for y in DM_var_err]   # DM only error at l=800

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
        ax.set_xlabel('Toys',fontsize=15)
        ax.legend(loc="upper right", fontsize=13)
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
    
    return (-1.0*mean)   


def get_fisher_var(var, lda_new, bins_, sample, scale_factors, x_label):
# lda_new: Rescale data to this lambda, to find fisher information. 
    m_chi=100;             # Mass of MC simulations. 
    l=100;                 # Lambda fof MC simulations. 
    rescale_factor=1.114;  # This is the ratio of cross-sections HepData/MG
    
    data=fl.read_data(sample);  # Read CSV files
    sig=data[0]; bkg_2=data[2]; jpt160=data[3];

    # Choose al data within some upper limit upper_lim:
    upper_lim=bins_[len(bins_)-1]
    sig=sig.query("{}{}{}".format((var),"<", upper_lim))
    jpt160=jpt160.query("{}{}{}".format((var),"<",upper_lim))
    bkg_2=bkg_2.query("{}{}{}".format((var),"<", upper_lim))

    fig, ax0 = plt.subplots(1,figsize=(8,6),sharex=True)
    ax0.set_title('$\sigma$ vs '+var+' for the '+sample+' region',fontsize=16)     
    ax0.text(0.03, 0.93, '$\Lambda_{EFT}$ = '+str(lda_new)+' GeV', fontsize=15, transform=ax0.transAxes)
    ax0.text(0.03, 0.88,  '$M_{\chi}$   = '+str(m_chi)+' GeV', fontsize=15, transform=ax0.transAxes)
        
    h=ax0.hist([bkg_2[var],jpt160[var],sig[var]], weights = [bkg_2['weight']
    *scale_factors[2],jpt160['weight']*(scale_factors[1]),sig['weight']
    *scale_factors[0]*pow(l/lda_new,6.0)], bins=bins_, color=['green','red','blue'], stacked=False,
    fill=False, histtype='step', label=[r'EWK $Z \rightarrow \nu \overline{\nu} + j j$ ', r'QCD $Z \rightarrow \nu \overline{\nu} + j$ '
                                        ,r'EFT  $ A \rightarrow \chi \overline{\chi}$ + jj'])

    ax0.set_ylabel('$\sigma(pb)$', fontsize=14)
    ax0.set_xlabel(var, fontsize=14)
    ax0.legend(loc="upper right", fontsize=12)
    
    errors=fl.get_error2(h, sample, var, lda_new, scale_factors)
    peaks_sig=[]; peaks_bkg_1=[]; peaks_bkg_2=[]; 
    for i in range(len(bins_)-1):
        peaks_sig.append(h[0][2][i])
        peaks_bkg_1.append(h[0][1][i])
        peaks_bkg_2.append(h[0][0][i])
        
    DM_var     = peaks_sig
    DM_var_err = errors[0] 
    SM_var     = [(x+y)*rescale_factor for x,y in zip (peaks_bkg_1,peaks_bkg_2)]
    SM_var_err = [np.sqrt(x**2+y**2)   for x,y in zip (errors[1],  errors[2])]
    
    # FISHER INFORMATION:
    
    # SM PREDICTION & EXPERIMENTAL ERROR:
    exp_frac_err=0.1;     # Use 10% expected experimental covariance for now...
    
    x          = SM_var                      # SM hypothesis
    sigma_x    = [exp_frac_err*x for x in SM_var]  # Expected Experimental Covariance.

    # BSM PREDICTION & (MADGRAPH) ERRORS:
    p          = [x+y for x,y in zip(SM_var, DM_var)]                  # SM+DM prediction at l=400.
    sigma_p    = [np.sqrt(x+y) for x,y in zip(SM_var_err, DM_var_err)] # Errors for SM+DM at l=400.


    # DM ONLY PREDICTION &  ERRORS:
    DM         = DM_var 
    sigma_DM   = DM_var_err

    BSM1       = [lda_new**6*DM for DM in DM_var] # DM at lambda=1: Used to calc. F.I.
    
    info_all_bins=[]
    
    for i in range(len(x)):
        info_bin_i=get_info(x[i], lda_new, p[i], BSM1[i], sigma_x[i], sigma_p[i], plot_dist=False)
        info_all_bins.append(info_bin_i)
    print('Information:', info_all_bins)
    
    plt.close()
    
    fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
    ax.set_title(x_label+' vs $\sigma$ for the '+sample+' region',fontsize=16)        

    ax.set_ylabel('$\sigma (pb)$', fontsize=14)
    ax.set_xlabel(x_label, fontsize=14)
              
    # Get Bin Centres and Widths:
    bin_centres=[]; bin_widths=[]
    for i in range(len(bins_)-1):
        ith_centre = 0.5*(bins_[i+1]-bins_[i])+bins_[i]
        ith_width  = (bins_[i+1]-bins_[i])
        bin_centres.append(ith_centre)
        bin_widths.append(ith_width)
    
    ax.bar(bin_centres,  x, fill=False, width=bin_widths, yerr=sigma_x,   ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction (with '+str(exp_frac_err*100)+'% error).')
    ax.bar(bin_centres, DM, fill=False, width=bin_widths, linestyle='--', yerr=sigma_DM, ecolor='red', edgecolor='red', capsize=4, label='MG5(LO) DM at $m_{\chi}$='+str(m_chi)+' $\Lambda_{EFT}$='+str(lda_new)+'.')

    ax_two=ax.twinx()
    ax_two.bar(bin_centres, info_all_bins, fill=True, alpha=0.2, width=bin_widths, facecolor='orange', label='Fisher Information')

    ax_two.set_ylabel('Fisher Information')
    ax_two.set_ylim(0)

    ax.legend(loc="upper left", fontsize=8)   
    ax_two.legend(loc="upper right", fontsize=8)
    plt.show()
    #plt.savefig(save_to+'/Fisher_Info_'+var+'_'+sample+'_m'+str(mchi)+'_l'+str(lda_new)+'.pdf')

#------------------------------------------------------------------------------

#Plot:
sum_of_weights=[29715.6,2354.22,6.33457];
cross_sections=[4.28E+04,2.26E+03,6.334571];
scale_f=[xs / sw for xs,sw in zip(cross_sections,sum_of_weights)];

# Pseudorapidity:
bins= [0, 0.5, 1, 1.5, 2, 4]
get_fisher_var('jeta1', 400, bins, 'vbf', scale_f, 'Leading jet $\eta$')

# Subleading Jet pT:
bins=[50, 75, 100, 150, 200, 300, 500]
get_fisher_var('jpt2', 400, bins, 'vbf', scale_f, 'Subleading jet $pT$')

# Leading Jet pT:
bins=[200, 250, 300, 400, 600, 1000]
get_fisher_var('jpt1',  400, bins, 'vbf', scale_f, 'Leading jet $pT$')


#-------------------------------------------------------------------------------------------
# PSEUDORAPIDITY PLOTS AND INFO:
if False:
    fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
    ax.set_title(r'$\sigma_{fid}$ vs Leading jet $\eta$ for the VBF region',fontsize=16)     
    ax.set_yscale('linear')

    ax.set_ylabel('$\sigma_{fid}$', fontsize=14)
    ax.set_xlabel('Leading jet $\eta$', fontsize=14)
              
    bin_centres=[0.25, 0.75, 1.25, 1.75, 3]  
    ax.bar(bin_centres,     x, fill=False, width=[0.5, 0.5, 0.5, 0.5, 2], yerr=sigma_x, ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction (with 10% error)')
    ax.bar(bin_centres, DM400, fill=False, width=[0.5, 0.5, 0.5, 0.5, 2], linestyle='--', yerr=sigma_DM400, ecolor='red', edgecolor='red', capsize=4, label='MG5(LO) DM at $m_{\chi}$=100, $\Lambda_{EFT}$=400')

    ax_two=ax.twinx()
    ax_two.bar(bin_centres, info_all_bins, fill=True, alpha=0.2, width=[0.5, 0.5, 0.5, 0.5, 2], facecolor='orange', label='Fisher Information')

    ax_two.set_ylabel('Fisher Information')
    ax_two.set_ylim(0)

    ax.legend(loc="upper left", fontsize=8)   
    ax_two.legend(loc="upper right", fontsize=8)
    #plt.savefig(save_to+'/Fisher_Info_rap_m'+str(mchi)+'_l400.pdf')


# JPT2 PLOTS AND INFO:
if False:
    fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
    ax.set_title(r'$\sigma$ vs subeading jet $pT$ for the VBF region',fontsize=16)     
    ax.set_yscale('linear')
    ax.set_ylim(0, 1.5)

    ax.set_ylabel('$\sigma$ (pb)', fontsize=14)
    ax.set_xlabel('Subleading jet $pT$ (GeV)', fontsize=14)
              
    bin_centres=[62.5, 87.5, 125, 175, 250, 400]  
    
    ax.bar(bin_centres,     x, fill=False, width=[25, 25, 50, 50, 100, 200], yerr=sigma_x, ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction (with 10% error)')
    ax.bar(bin_centres, DM400, fill=False, width=[25, 25, 50, 50, 100, 200], linestyle='--', yerr=sigma_DM400, ecolor='red', edgecolor='red', capsize=4, label='MG5(LO) DM at $m_{\chi}$=100, $\Lambda_{EFT}$=400')

    ax_two=ax.twinx()
    ax_two.bar(bin_centres, info_all_bins, fill=True, alpha=0.2, width=[25, 25, 50, 50, 100, 200], facecolor='orange', label='Fisher Information')

    ax_two.set_ylabel('Fisher Information')
    ax_two.set_ylim(0)

    ax.legend(loc="upper left", fontsize=8)   
    ax_two.legend(loc="upper right", fontsize=8)
    # plt.savefig(save_to+'/Fisher_Info_jpt2_m'+str(mchi)+'_l800.pdf')


# JPT1 PLOTS AND INFO:
if False:
    fig, ax = plt.subplots(1,figsize=(9,6),sharex=True)
    ax.set_title(r'$\sigma$ vs leading jet $pT$ for the $\geq$ 1 jet region',fontsize=16)     
    ax.set_yscale('linear')

    ax.set_ylabel('$\sigma$ (pb)', fontsize=14)
    ax.set_xlabel('Leading jet $pT$ (GeV)', fontsize=14)
          
    bin_centres=[225, 275, 350, 500, 800]  

    ax.bar(bin_centres,     x, fill=False, width=[50, 50, 100, 200, 400], yerr=sigma_x, ecolor='blue', edgecolor='blue', capsize=4, label='MG5(LO) SM prediction (with 10% error)')
    ax.bar(bin_centres, DM400, fill=False, width=[50, 50, 100, 200, 400], linestyle='--', yerr=sigma_DM400, ecolor='red', edgecolor='red', capsize=4, label='MG5(LO) DM at $m_{\chi}$=100, $\Lambda_{EFT}$=400')

    ax_two=ax.twinx()
    ax_two.bar(bin_centres, info_all_bins, fill=True, alpha=0.2, width=[50, 50, 100, 200, 400], facecolor='orange', label='Fisher Information')

    ax_two.set_ylabel('Fisher Information')
    ax_two.set_ylim(0)

    ax.legend(loc="upper left", fontsize=8)   
    ax_two.legend(loc="upper right", fontsize=8)
    plt.savefig(save_to+'/Fisher_Info_jpt1_MONO_m'+str(mchi)+'_l400.pdf')









