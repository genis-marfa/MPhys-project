import pandas
import math 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import sys
import pickle

file_path = '/Users/Genis/mphys/Rivet/DM/HnoE/'
path_2='/Users/Genis/mphys/Rivet/DM/jpt1_testing/'
samples=['mono','vbf']
eps=10e-6
m_chi=100; l=100;
num_bins=20;

def read_data(sample): 
    signal=[pandas.read_csv(file_path+"DM_l100_m100_50k_"+sample+".csv")]
    background_1 = [pandas.read_csv(file_path+"QCD_Znunu_l100_mchi100_100k_"+sample+".csv")]
    background_2 = [pandas.read_csv(file_path+"EWK_l100_mchi100_50k_"+sample+".csv")]
    jpt1_60= [pandas.read_csv(path_2+"QCD_jpt1_60_100k_"+sample+".csv")]
    
    sig=signal[0]; bkg_1=background_1[0]; bkg_2=background_2[0]; jpt160=jpt1_60[0]
    sig.name='sig'; bkg_1.name='bkg_1'; bkg_2.name='bkg_2';
    
    return [sig, bkg_1, bkg_2, jpt160]

read_data('vbf')
jpt_sample=[];

# ______________________________________________________________________________
# ==============================================================================
# HISTOPLOT: PLOT HISTOGRAMS PAGE 16, FIGURE 4, (a-d) at some lambda = lda_new:
# ==============================================================================

def histoplot(lda_new, scale_factors, plot, show=True):
# ==========================================
# Fig. A: sigma vs met for mono jet region.
# ==========================================   
    if plot=='a':
        data=read_data('mono')
        sig=data[0]; bkg_2=data[2]; jpt160=data[3];
    
        sig=sig.query("{}{}{}".format(('met'),"<",1400))
        jpt160=jpt160.query("{}{}{}".format(('met'),"<",1400))
        bkg_2=bkg_2.query("{}{}{}".format(('met'),"<",1400))
    
        fig, ax0 = plt.subplots(1,figsize=(8,6),sharex=True)
        ax0.set_title('(a): $\sigma_{fid}$ vs $p_T^{miss}$ for $\geq$ 1 jet region',fontsize=16)     
        ax0.text(0.03, 0.93, '$\Lambda_{EFT}$ = '+str(lda_new)+' GeV', fontsize=15, transform=ax0.transAxes)
        ax0.text(0.03, 0.88,  '$M_{\chi}$   = '+str(m_chi)+' GeV', fontsize=15, transform=ax0.transAxes)
        ax0.set_yscale('log')
    
        h=ax0.hist([bkg_2['met'],jpt160['met'],sig['met']], weights = [bkg_2['weight']
        *scale_factors[2],jpt160['weight']*(scale_factors[1]),sig['weight']
        *scale_factors[0]*pow(l/lda_new,6.0)], bins=[200, 250, 300, 350, 500, 700, 1000, 1400], color=['green','red','blue'], stacked=False,
        fill=False, histtype='step', label=[r'EWK $Z \rightarrow \nu \overline{\nu} + j j$ ', r'QCD $Z \rightarrow \nu \overline{\nu} + j$ '
                                 ,r'EFT  $ A \rightarrow \chi \overline{\chi}$ + jj'])

        ax0.set_ylabel('$\sigma_{fid}$', fontsize=14)
        ax0.set_xlabel('$p_T^{miss}$', fontsize=14)
    
        peaks_sig=[]; peaks_bkg_1=[]; peaks_bkg_2=[]; 
        for i in range(7):
            if h[0][2][i] > eps:
                peaks_sig.append(h[0][2][i])
            if h[0][1][i] > eps:
                peaks_bkg_1.append(h[0][1][i])
            if h[0][0][i] > eps:
                peaks_bkg_2.append(h[0][0][i])
        
        y_max=400.*max([max(peaks_sig),max(peaks_bkg_1),max(peaks_bkg_2)])
        y_min=min([min(peaks_sig),min(peaks_bkg_1),min(peaks_bkg_2)])/5.
        ax0.set_xlim(200, 1400)
        ax0.set_ylim(y_min, y_max)
        ax0.legend(loc="upper right", fontsize=12)
    
        if show==True:
            plt.show()          
        return h

# ==========================================
# Fig. B: sigma vs met for vbf region.
# ==========================================             
    elif plot=='b':
        data=read_data('vbf')
        sig=data[0]; bkg_2=data[2]; jpt160=data[3];
    
        sig=sig.query("{}{}{}".format(('met'),"<",1400))
        jpt160=jpt160.query("{}{}{}".format(('met'),"<",1400))
        bkg_2=bkg_2.query("{}{}{}".format(('met'),"<",1400))
    
        fig, ax = plt.subplots(1,figsize=(8,6),sharex=True)
        ax.set_title('(b): $\sigma_{fid}$ vs $p_T^{miss}$ for VBF region: ',fontsize=16)   
        ax.text(0.03, 0.93, '$\Lambda_{EFT}$ = '+str(lda_new)+' GeV', fontsize=15, transform=ax.transAxes)
        ax.text(0.03, 0.88,  '$M_{\chi}$   = '+str(m_chi)+' GeV', fontsize=15, transform=ax.transAxes)
        ax.set_yscale('log')
    
        h_2=ax.hist([bkg_2['met'],jpt160['met'],sig['met']], weights = [bkg_2['weight']
        *scale_factors[2],jpt160['weight']*(scale_factors[1]),sig['weight']
        *scale_factors[0]*pow(l/lda_new,6.0)], bins=[200, 250, 300, 350, 500, 700, 1400], color=['green','red','blue'], stacked=False,
        fill=False, histtype='step', label=[r'EWK $Z \rightarrow \nu \overline{\nu} + j j$ ', r'QCD $Z \rightarrow \nu \overline{\nu} + j$ '
                                 ,r'EFT  $ A \rightarrow \chi \overline{\chi}$ + jj'])

        ax.set_ylabel('$\sigma_{fid}$', fontsize=14)
        ax.set_xlabel('$p_T^{miss}$', fontsize=14)
    
        peaks_sig=[]; peaks_bkg_1=[]; peaks_bkg_2=[]; 
        for i in range(6):
            if h_2[0][2][i] > eps:
                peaks_sig.append(h_2[0][2][i])
            if h_2[0][1][i] > eps:
                peaks_bkg_1.append(h_2[0][1][i])
            if h_2[0][0][i] > eps:
                peaks_bkg_2.append(h_2[0][0][i])
        y_max=400.*max([max(peaks_sig),max(peaks_bkg_1),max(peaks_bkg_2)])
        y_min=min([min(peaks_sig),min(peaks_bkg_1),min(peaks_bkg_2)])/5.
        ax.set_xlim(200, 1400)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc="upper right", fontsize=12)
    
        if show==True:
            plt.show()
        
        return h_2

# ==========================================
# Fig. C: sigma vs mjj for vbf region.
# ==========================================     
    elif plot=='c':
        data=read_data('vbf')
        sig=data[0]; bkg_2=data[2]; jpt160=data[3];
    
        sig=sig.query("{}{}{}".format(('mjj'),"<",4000))
        jpt160=jpt160.query("{}{}{}".format(('mjj'),"<",4000))
        bkg_2=bkg_2.query("{}{}{}".format(('mjj'),"<",4000))
    
        fig, ax2 = plt.subplots(1,figsize=(8,6),sharex=True)
        ax2.set_title('(c): $\sigma_{fid}$ vs $m_{jj}$ for VBF region:',fontsize=16)     
        ax2.text(0.03, 0.93, '$\Lambda_{EFT}$ = '+str(lda_new)+' GeV', fontsize=15, transform=ax2.transAxes)
        ax2.text(0.03, 0.88,  '$M_{\chi}$   = '+str(m_chi)+' GeV', fontsize=15, transform=ax2.transAxes)
        ax2.set_yscale('log')
    
        h_3=ax2.hist([bkg_2['mjj'],jpt160['mjj'],sig['mjj']], weights = [bkg_2['weight']
        *scale_factors[2],jpt160['weight']*(scale_factors[1]),sig['weight']
        *scale_factors[0]*pow(l/lda_new,6.0)], bins=[200,400,600,1000,2000,4000], color=['green','red','blue'], stacked=False,
        fill=False, histtype='step', label=[r'EWK $Z \rightarrow \nu \overline{\nu} + j j$ ', r'QCD $Z \rightarrow \nu \overline{\nu} + j$ '
                                 ,r'EFT  $ A \rightarrow \chi \overline{\chi}$ + jj'])

        ax2.set_ylabel('$\sigma_{fid}$', fontsize=14)
        ax2.set_xlabel('$m_{jj}$', fontsize=14)
    
        peaks_sig=[]; peaks_bkg_1=[]; peaks_bkg_2=[]; 
        for i in range(5):
            if h_3[0][2][i] > eps:
                peaks_sig.append(h_3[0][2][i])
            if h_3[0][1][i] > eps:
                peaks_bkg_1.append(h_3[0][1][i])
            if h_3[0][0][i] > eps:
                peaks_bkg_2.append(h_3[0][0][i])
        y_max=400.*max([max(peaks_sig),max(peaks_bkg_1),max(peaks_bkg_2)])
        y_min=min([min(peaks_sig),min(peaks_bkg_1),min(peaks_bkg_2)])/5.
        ax2.set_xlim(200, 4000)
        ax2.set_ylim(y_min, y_max)
        ax2.legend(loc="upper right", fontsize=12)
    
        if show==True:
            plt.show()   
        return h_3
# ==========================================
# Fig. D: sigma vs dphi_jj for vbf region.
# ==========================================     
    elif plot=='d':
        data=read_data('vbf')
        sig=data[0]; bkg_2=data[2]; jpt160=data[3];
    
        sig=sig.query("{}{}{}".format(('dphi'),"<",np.pi))
        jpt160=jpt160.query("{}{}{}".format(('dphi'),"<",np.pi))
        bkg_2=bkg_2.query("{}{}{}".format(('dphi'),"<",np.pi))
    
        fig, ax3 = plt.subplots(1,figsize=(8,6),sharex=True)
        ax3.set_title('(d): $\sigma_{fid}$ vs $\Delta \phi_{jj}$ for VBF region:',fontsize=16)     
        ax3.text(0.03, 0.93, '$\Lambda_{EFT}$ = '+str(lda_new)+' GeV', fontsize=15, transform=ax3.transAxes)
        ax3.text(0.03, 0.88,  '$M_{\chi}$   = '+str(m_chi)+' GeV', fontsize=15, transform=ax3.transAxes)
        ax3.set_yscale('log')
    
        h_4=ax3.hist([bkg_2['dphi'],jpt160['dphi'],sig['dphi']], weights = [bkg_2['weight']
        *scale_factors[2],jpt160['weight']*(scale_factors[1]),sig['weight']
        *scale_factors[0]*pow(l/lda_new,6.0)], bins=[0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6,np.pi], color=['green','red','blue'], stacked=False,
        fill=False, histtype='step', label=[r'EWK $Z \rightarrow \nu \overline{\nu} + j j$ ', r'QCD $Z \rightarrow \nu \overline{\nu} + j$ '
                                 ,r'EFT  $ A \rightarrow \chi \overline{\chi}$ + jj'])

        ax3.set_ylabel('$\sigma_{fid}$', fontsize=14)
        ax3.set_xlabel('$\Delta \phi_{jj}$', fontsize=14)
    
        peaks_sig=[]; peaks_bkg_1=[]; peaks_bkg_2=[]; 
        for i in range(6):
            if h_4[0][2][i] > eps:
                peaks_sig.append(h_4[0][2][i])
            if h_4[0][1][i] > eps:
                peaks_bkg_1.append(h_4[0][1][i])
            if h_4[0][0][i] > eps:
                peaks_bkg_2.append(h_4[0][0][i])
        y_max=400.*max([max(peaks_sig),max(peaks_bkg_1),max(peaks_bkg_2)])
        y_min=min([min(peaks_sig),min(peaks_bkg_1),min(peaks_bkg_2)])/5.
        ax3.set_xlim(0, np.pi)
        ax3.set_ylim(y_min, y_max)
        ax3.legend(loc="upper right", fontsize=12)
    
        if show==True:
            plt.show()    
        return h_4
    else:
        raise ValueError("Please specify plot: a, b, c or d")
#-------------------------------------------------------------------------------------------#
def get_error2(h, sample, var, lda_new, scale_factors, summ=True):
    data=read_data(sample)
    sig=data[0]; bkg_2=data[2]; jpt160=data[3];
    sig.name='sig'; bkg_2.name='bkg_2'; jpt160.name='bkg_1';
    

    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100*0/3.))
    sys.stdout.flush()
    
    sigma_sig=[]; sum_weight_square=0; N=[]; k=0;
    for i in range(len(h[1])-1):
        for row in range(len(sig)):
            if (sig.iloc[row][var])>(h[1][i]) and (sig.iloc[row][var])<(h[1][i+1]+eps):
                weight=sig.iloc[row]['weight'];
                weight_sq=weight*weight;
                sum_weight_square+=weight_sq;  
                k+=1;
        N.append(k); k=0;                
        sigma_sig.append(math.sqrt(sum_weight_square)*scale_factors[0]*pow(100/lda_new,6.0));
        sum_weight_square=0;
    
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100*1./3.))
    sys.stdout.flush()
    
    sigma_jpt160=[]; sum_weight_square=0; N_2=[]; k=0;
    for i in range(len(h[1])-1):
        for row in range(len(jpt160)):
            if (jpt160.iloc[row][var])>(h[1][i]) and (jpt160.iloc[row][var])<(h[1][i+1]+eps):
                weight=jpt160.iloc[row]['weight'];
                weight_sq=weight*weight;
                sum_weight_square+=weight_sq; 
                k+=1 
        N_2.append(k); k=0;                   
        sigma_jpt160.append(math.sqrt(sum_weight_square)*scale_factors[1]);
        sum_weight_square=0;
        
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100*2./3.))
    sys.stdout.flush()

    sigma_bkg2=[]; sum_weight_square=0; N_3=[]; k=0;
    for i in range(len(h[1])-1):
        for row in range(len(bkg_2)):
            if (bkg_2.iloc[row][var])>(h[1][i]) and (bkg_2.iloc[row][var])<(h[1][i+1]+eps):
                weight=bkg_2.iloc[row]['weight'];
                weight_sq=weight*weight;
                sum_weight_square+=weight_sq;     
                k+=1 
        N_3.append(k); k=0;  
        sigma_bkg2.append(math.sqrt(sum_weight_square)*scale_factors[2]);
        sum_weight_square=0;
    
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100*3./3.))
    sys.stdout.flush()
    print('.')
    
    rel_err_Poi=np.sqrt(N)/(N); rel_err_MC=sigma_sig/h[0][2]
    print('Done!')
# (Optionally): Make a small summary report:       
    if summ==True:
        print('Error calculation returned:')
        print('====================================================================')
        print('                               DM EFT:                              ')
        print('====================================================================')
        print('Relative error assuming Poisson Statistics at each bin:'); print(np.around(rel_err_Poi, decimals=4))
        print('Events into each bin'); print(N);
        print('Relative error for MC weighted events:'); print(np.around(rel_err_MC, decimals=4))
        print('Sigma:', np.around(sigma_sig,decimals=5)) 
        print('====================================================================')
        print(r'                          QCD Z -> nu nu:                         ')
        print('====================================================================')     
        rel_err_Poi=np.sqrt(N_2)/(N_2); rel_err_MC=sigma_jpt160/h[0][1]        
        print('Relative error assuming Poisson Statistics at each bin:'); print(np.around(rel_err_Poi, decimals=4))
        print('Events into each bin'); print(N_2);
        print('Relative error for MC weighted events:'); print(np.around(rel_err_MC, decimals=4))  
        print('Sigma:', np.around(sigma_jpt160,decimals=3));
        print('====================================================================')
        print(r'                          EWK Z -> nu nu:                         ')
        print('====================================================================')   
        rel_err_Poi=np.sqrt(N_3)/(N_3); rel_err_MC=sigma_bkg2/h[0][0]        
        print('Relative error assuming Poisson Statistics at each bin:'); print(np.around(rel_err_Poi, decimals=4))
        print('Events into each bin'); print(N_3);
        print('Relative error for MC weighted events:'); print(np.around(rel_err_MC, decimals=4))   
        print('Sigma:', np.around(sigma_bkg2,decimals=4)); 
        print('--------------------------------------------------------------------')     
    
    return [sigma_sig, sigma_jpt160, sigma_bkg2]
#-------------------------------------------------------------------------------------------#
def get_ylimits(histo,k,var):
    peaks_sig=[]; peaks_bkg_1=[]; peaks_bkg_2=[]; 
    for i in range(len(histo[1])-1):
        if histo[0][2][i] > eps:
            peaks_sig.append(histo[0][2][i])
        if histo[0][1][i] > eps:
            peaks_bkg_1.append(histo[0][1][i])
        if histo[0][0][i] > eps:
            peaks_bkg_2.append(histo[0][0][i])
    
    y_max=k*20*max([max(peaks_sig),max(peaks_bkg_1),max(peaks_bkg_2)])
    y_min=min([min(peaks_sig),min(peaks_bkg_1),min(peaks_bkg_2)])/k
    if var=='dphi':
      y_max=k*15*max([max(peaks_sig),max(peaks_bkg_1),max(peaks_bkg_2)])  
    return[y_min,y_max]
#-------------------------------------------------------------------------------------------#
def errorbar_plot(plot, scale_factors,li,lf,linc, show=True, summ=True):
    if plot=='a':
        title='(a): $\sigma_{fid}$ vs $p_T^{miss}$ for $\geq$ 1 jet region';
        xlabel='$pT^{miss}$'; sample='mono';  var='met';
        bin_widths=[50, 50, 50, 150, 200, 300, 400]
    elif plot=='b':
        title='(b): $\sigma_{fid}$ vs $p_T^{miss}$ for VBF region: ';
        xlabel='$pT^{miss}$';  sample='vbf'; var='met';
        bin_widths=[50, 50, 50, 150, 200, 700]
    elif plot=='c':
        title='(c): $\sigma_{fid}$ vs $m_{jj}$ for VBF region:';
        xlabel='$m_{jj}$'; sample='vbf'; var='mjj';
        bin_widths=[200, 200, 400, 1000, 2000]
    elif plot=='d':
        title='(d): $\sigma_{fid}$ vs $\Delta \phi_{jj}$ for VBF region:';
        xlabel='$\Delta \phi_{jj}$'; sample='vbf'; var='dphi'
        bin_widths=np.pi/6
    else: 
       raise ValueError("Please specify plot: a, b, c or d") 
        
    n0=0; n1=1;
    for lda_new in range (li,lf+1,linc):
        histo=histoplot(lda_new, scale_factors, plot, show) 
    
    # First get bin centres:
        bin_centres=[];
        for i in range(len(histo[1])-1):
            bin_centres.append((histo[1][i]+histo[1][i+1])/2)
    
    # get sigma:
        if lda_new == li:             
            sigmas=get_error2(histo, sample, var, lda_new, scale_factors, summ)
            sigma_sig=sigmas[0]; sigma_jpt160=sigmas[1]; sigma_bkg_2=sigmas[2]; 
        
    # plot:
        plt.close()
        fig, ax= plt.subplots(1,figsize=(8,6),sharex=True)
        ax.bar(bin_centres,histo[0][2],width=bin_widths, yerr=sigma_sig,
           edgecolor='blue', fill=False, capsize=4, ecolor='blue', label=r'EFT  $ A \rightarrow \chi \overline{\chi}$ + jj')
        ax.bar(bin_centres,histo[0][1],width=bin_widths, edgecolor='red', 
            yerr=sigma_jpt160, fill=False, capsize=4, ecolor='red', label=r'QCD $Z \rightarrow \nu \overline{\nu} + j$ ')
        ax.bar(bin_centres,histo[0][0],width=bin_widths, yerr=sigma_bkg_2, 
           edgecolor='green', fill=False, capsize=4, ecolor='green', label=r'EWK $Z \rightarrow \nu \overline{\nu} + j j$ ')
        ax.text(0.03, 0.93, '$\Lambda_{EFT}$ = '+str(lda_new)+' GeV', fontsize=15, transform=ax.transAxes)
        ax.text(0.03, 0.88,  '$M_{\chi}$   = '+str(m_chi)+' GeV', fontsize=15, transform=ax.transAxes)
        
        ax.set_yscale('log')      
        limits=get_ylimits(histo,10,var)
        ax.set_ylim(limits[0],limits[1])
        ax.set_ylabel('$\sigma_{fid}$', fontsize=14)
        ax.legend(loc="upper right", fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
           
        save_to='/Users/Genis/mphys/Rivet/DM/HNoE/plots/plots_nov_22'; 
        plt.savefig(save_to+'/Plot '+plot+'. '+var+' l'+str(lda_new)+' mchi'+str(m_chi)+'.pdf')
        plt.show()  
        
        for i in range (len(sigma_sig)):
            sigma_sig[i]=sigma_sig[i]*pow(li+(n0*linc),6.0)/pow(li+(n1*linc),6.0)      
        n0+=1; n1+=1;
#______________________________________________________________________________        
def hepdata_compare(plot, scale_factors, show=True, summ=True):
    if plot=='a':   
        title='(a): $\sigma_{fid}$ vs $p_T^{miss}$ for $\geq$ 1 jet region';
        xlabel='$pT^{miss}$'; ylabel=r'$\frac{\sigma_{fid}}{Bin \ Width}$ [$fb$ $GeV^{-1}$]'; sample='mono'; var='met'; 
        SM_HepData=[155.9, 62.8, 27.4, 7.73, 1.06, 0.126, 0.011]
        err_HepData=[2.029e+01, 9.645e+00, 4.956e+00, 1.359e+00, 2.070e-01, 2.257e-02, 1.925e-03]
        bin_widths=[50, 50, 50, 150, 200, 300, 400]
    elif plot=='b':
        title='(b): $\sigma_{fid}$ vs $p_T^{miss}$ for VBF region: ';
        xlabel='$pT^{miss}$'; ylabel=r'$\frac{\sigma_{fid}}{Bin \ Width}$ [$fb$ $GeV^{-1}$]'; sample='vbf'; var='met';
        SM_HepData=[33.6, 14.93, 7.0794, 2.2748, 0.38495, 0.0271]
        err_HepData=[3.161e+00, 1.668e+00, 8.900e-01, 3.007e-01, 6.020e-02, 3.929e-03];
        bin_widths=[50, 50, 50, 150, 200, 700]
    elif plot=='c':
        title='(c): $\sigma_{fid}$ vs $m_{jj}$ for VBF region:';
        xlabel='$m_{jj}$'; ylabel=r'$\frac{\sigma_{fid}}{Bin \ Width}$ [$fb$ $GeV^{-1}$]'; sample='vbf'; var='mjj';
        SM_HepData=[8.834, 3.699, 1.208, 0.1994, 0.0141]
        err_HepData=[9.642000e-01, 3.981500e-01, 1.247000e-01, 1.942000e-02, 1.435000e-03]
        bin_widths=[200, 200, 400, 1000, 2000]
    elif plot=='d':
        title='(d): $\sigma_{fid}$ vs $\Delta \phi_{jj}$ for VBF region:';
        xlabel='$\Delta \phi_{jj}$'; ylabel=r'$\frac{\sigma_{fid}}{Bin \ Width}$ [$fb$ $rad^{-1}$]'; sample='vbf'; var='dphi';
        SM_HepData=[630.2, 717.7, 1.066E+03, 1.409E+03, 1.297E+03, 7.034E+02]
        err_HepData=[6.063803e+01, 6.946158e+01, 1.143815e+02, 1.636176e+02, 1.476130e+02, 6.953798e+01]
        bin_widths=[np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6]
           
    h=histoplot(100, scale_factors, plot, show)
    sigmas=get_error2(h, sample, var, 100, scale_factors, summ)
    sigma_qcd=sigmas[1]; sigma_ewk=sigmas[2];
    
    BSM_pred=h[0][2];           BSM_pred[:]= [x*1000 for x in BSM_pred];
    # Normalise BSM:
    BSM_pred[:]=[x/y for x,y in zip(BSM_pred, bin_widths)]


    SM_pred= (h[0][0]+h[0][1]); SM_pred[:]=  [x*1000 for x in SM_pred];
    
    # Un-normalise HepData:
    SM_HepData=[x*y for x,y in zip(SM_HepData, bin_widths)] 
    # Find Rescale Factor:
    rescale_factor=np.sum(SM_HepData)/np.sum(SM_pred)
    print('Rescale Factor:', rescale_factor);
    # Renormalise HepData
    SM_HepData=[x/y for x,y in zip(SM_HepData, bin_widths)]
    #Reweighing:
    SM_pred[:]=[x*rescale_factor/y for x,y in zip(SM_pred, bin_widths)]
    
    sigma_quad=[np.sqrt(x**2+y**2) for x,y in zip(sigma_qcd,sigma_ewk)] # add errors in quadrature.
    
    bin_centres=[];
    for i in range(len(h[1])-1):
        bin_centres.append((h[1][i]+h[1][i+1])/2)
    
    
    plt.close()
    fig, ax= plt.subplots(1,figsize=(8,6),sharex=True)
    ax.bar(bin_centres,SM_pred,width=bin_widths, yerr=sigma_quad,
           edgecolor='blue', fill=False, capsize=4, ecolor='blue', label=r'MG5(LO) SM prediction:')
    ax.bar(bin_centres,SM_HepData, width=bin_widths,  yerr=err_HepData,
           fill=False, edgecolor='red', capsize=4,ecolor='red', label=r'HepData prediction' )
    ax.set_ylabel(ylabel, fontsize=14)
    
    if plot=='d':
        ax.legend(loc="upper left", fontsize=12)
    else:
        ax.legend(loc="upper right", fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_yscale('log')
    save_to='/Users/Genis/mphys/Rivet/DM/HNoE/plots/plots_nov_22'; 
    plt.savefig(save_to+'/Plot '+plot+'.m'+str(m_chi)+' '+var+'SM_Comparison.pdf')
    
    # Residuals:
    fig, ax2= plt.subplots(1,figsize=(8,2),sharex=True)
    SM_pred_rat=[x/y for x,y in zip(SM_pred, SM_HepData)]
    SM_HepData_rat=[x/y for x,y in zip(SM_HepData, SM_HepData)]
    
    SM_rat_err=[np.sqrt((x1/y1)**2+(x2/y2)**2)*z for x1,y1,x2,y2,z in zip(sigma_quad,SM_pred,err_HepData,SM_HepData, SM_pred_rat)]
    
    ax2.errorbar(bin_centres,SM_pred_rat, yerr=SM_rat_err,
           fmt='ob', capsize=4, ecolor='blue', label=r'MG5(LO) SM prediction:')
    ax2.bar(bin_centres,SM_HepData_rat, width=bin_widths,
           fill=False, edgecolor='red', label=r'HepData prediction' )
    ax2.set_yscale('linear')
    ax2.set_xlabel(xlabel, fontsize=14)
    
    plt.savefig(save_to+'/Residuals_Plot '+plot+'.m'+str(m_chi)+' '+var+'SM_Comparison.pdf')
    plt.show()       
#______________________________________________________________________________
def rescale_data(lda_new, scale_factors, plot, replot=True, summ1=True, summ2=True): 
    h=histoplot(lda_new, scale_factors, plot, False); plt.close()
    # Load ATLAS DATA:
    inputs = pickle.load(open("paper_values_and_DMEFT_D7a_m10_l400_prediction.pickle", "rb"))
    meas=inputs["meas_values"]  
    
    if plot=='a':   
        title='(a): $R_{miss}$ vs $p_T^{miss}$ for $\geq$ 1 jet region';
        xlabel='$pT^{miss}$'; ylabel=ylabel=r'$R_{miss}$'; sample='mono'; var='met'; 
        SM_HepData=   [155.9, 62.8, 27.4, 7.73, 1.06, 0.126, 0.011]
        err_HepData=  [4.244000e-01, 2.300000e-01, 5.560000e-02, 1.786667e-02, 1.800000e-03, 5.000000e-04, 5.000000e-05]
        denominators= [2.096440e+01, 8.930, 4.0748, 1.1924, 1.708e-01, 2.12e-02, 1.85e-03]     # Differential Cross sections for (ll+jj).
        err_denom=    [7.240000e-02, 3.920000e-02, 1.080000e-02, 3.333333e-03, 6.500000e-04, 3.333333e-04, 2.500000e-05] # Errors on measurements. 
        bin_widths=   [50, 50, 50, 150, 200, 300, 400]
        meas=meas[0:7]; meas_stat= [(0.13,0.1,0.14,0.13,0.28,0.64,1.4), [0.13, 0.21, 0.27, 0.27, 0.54, 1.52, 3.64]]
    elif plot=='b':
        title='(b): $R_{miss}$ vs $p_T^{miss}$ for VBF region: ';
        xlabel='$pT^{miss}$'; ylabel=r'$R_{miss}$'; sample='vbf'; var='met';
        SM_HepData=   [33.6, 14.93, 7.0794, 2.2748, 0.38495, 0.0271]
        err_HepData=  [1.694000e-01, 9.400000e-02, 2.660000e-02, 8.133333e-03, 1.500000e-03, 2.142857e-04];
        denominators= [4.5316, 2.1202, 1.0602, 3.555333e-01, 6.255e-02, 4.628571e-03]
        err_denom =   [2.880000e-02, 1.740000e-02, 5.200000e-03, 1.666667e-03, 4.500000e-04, 1.285714e-04];
        bin_widths=   [50, 50, 50, 150, 200, 700]
        meas=meas[7:13]; meas_stat= [(0.27,0.21,0.28,0.22,0.42, 0.94), [0.27, 0.42, 0.54, 0.43, 0.79, 2.50]]
    elif plot=='c':
        title='(c): $R_{miss}$ vs $m_{jj}$ for VBF region:';
        xlabel='$m_{jj}$'; ylabel=r'$R_{miss}$'; sample='vbf'; var='mjj';
        SM_HepData=   [8.834, 3.699, 1.208, 0.1994, 0.0141]
        err_HepData=  [3.730000e-02, 2.300000e-02, 9.600000e-03, 2.190000e-03, 4.100000e-04]
        denominators= [1.24525, 5.215e-01, 1.7255e-01, 2.824e-02, 1.98e-03]
        err_denom =   [6.400000e-03, 4.000000e-03, 1.750000e-03, 4.200000e-04, 7.000000e-05]
        bin_widths=   [200, 200, 400, 1000, 2000]
        meas=meas[13:18]; meas_stat= [(0.27,0.19,0.24,0.41,1.18), [0.27, 0.36, 0.44, 0.78, 2.60]];
    elif plot=='d':
        title='(d): $R_{miss}$ vs $\Delta \phi_{jj}$ for VBF region:';
        xlabel='$\Delta \phi_{jj}$'; ylabel=r'$R_{miss}$'; sample='vbf'; var='dphi';
        SM_HepData=   [630.2, 717.7, 1.066E+03, 1.409E+03, 1.297E+03, 7.034E+02]
        err_HepData=  [6.111550e+00, 6.378930e+00, 7.906818e+00, 9.148226e+00, 8.919043e+00, 6.703606e+00]
        denominators= [9.541657e+01, 1.043547e+02, 1.536673e+02, 1.984726e+02, 1.807491e+02, 8.760525e+01]
        err_denom =   [1.126817e+00, 1.241409e+00, 1.336902e+00, 1.566085e+00, 1.566085e+00, 1.012225e+00]        
        bin_widths=   [np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6]       
        meas=meas[18:24]; meas_stat= [(0.46,0.25,0.22,0.19,0.25, 0.33), [0.46, 0.48, 0.45, 0.36, 0.47, 0.59]];

    sigmas=get_error2(h, sample, var, lda_new, scale_factors, summ1)
    sigma_DM=sigmas[0]; sigma_qcd=sigmas[1]; sigma_ewk=sigmas[2];
    
    # DM prediction adjustments:
    BSM_pred=h[0][2];           
    BSM_pred[:]= [x*1000 for x in BSM_pred]; # Convert pb to fb
    BSM_pred[:]=[x/y for x,y in zip(BSM_pred, bin_widths)] # Normalise to bin width

    # SM prediction adjustments
    SM_pred= (h[0][0]+h[0][1]); 
    SM_pred[:]= [x*1000 for x in SM_pred];
    SM_HepData=[x*y for x,y in zip(SM_HepData, bin_widths)] # Un-normalise HepData
    rescale_factor=np.sum(SM_HepData)/np.sum(SM_pred)       # Find Rescale Factor
    
    
    print('HEPDATA CROSS-SECTION', np.sum(SM_HepData))
    print('MG5 CROSS-SECTION', np.sum(SM_pred))
    
    print('Rescale Factor:', rescale_factor);               # Print rescale factor
    SM_HepData=[x/y for x,y in zip(SM_HepData, bin_widths)] # Renormalise HepData
    
    # Reweighing and normalising:
    SM_pred[:]=[x*rescale_factor/y for x,y in zip(SM_pred, bin_widths)]
    
    # Construct R_miss:
    Rmiss_SM  = [x/y for x,y in zip(SM_pred, denominators)]
    Rmiss_DM = [x/y for x,y in zip(BSM_pred, denominators)]
    Rmiss_sum = [x+y for x,y in zip(Rmiss_SM, Rmiss_DM)]
    
    
    print('Propagating error...')
    # Combine in quadrature error of two backgrounds:
    sigma_quad=[np.sqrt(x**2+y**2) for x,y in zip(sigma_qcd,sigma_ewk)] # add errors in quadrature.
    #print('BSM Unscaled', BSM_pred_unscaled, 'Err BSM uscaled:', sigma_quad)
    
    # Propagate error on correlated measurements of cs_tot_HepData / cs_tot_MadGraph
    
    # Get error on R_miss : numerator / denominators:
    err_Rmiss_SM=  [np.sqrt((x1/y1)**2+(x2/y2)**2)*z for x1,y1,x2,y2,z in zip(sigma_quad,SM_pred,err_denom, denominators,  Rmiss_SM)]
    err_Rmiss_DM= [np.sqrt((x1/y1)**2+(x2/y2)**2)*z for x1,y1,x2,y2,z in zip(sigma_DM,BSM_pred,err_denom, denominators, Rmiss_DM)]        
    err_Rmiss_quad=[np.sqrt(x**2+y**2) for x,y in zip(err_Rmiss_SM,err_Rmiss_DM)] # add errors in quadrature.
          
    print('Done!')
    if summ2==True:
        print('====================================================================')
        print('                          Standard Model:                           ')
        print('====================================================================')
        print('Rmiss prediction:')
        Rmiss_3dp=[np.round(x,3) for x in Rmiss_SM]
        Rmiss_err_3dp=[np.round(x,3) for x in err_Rmiss_SM]
        print(Rmiss_3dp)
        print('Error Rmiss:')
        print(Rmiss_err_3dp)
        print('====================================================================')
        print('                         EFT Dark Matter:                           ')
        print('====================================================================')
        print('Rmiss prediction:')
        Rmiss_3dp=[np.round(x,3) for x in Rmiss_DM]
        Rmiss_err_3dp=[np.round(x,3) for x in err_Rmiss_DM]
        print(Rmiss_3dp)
        print('Error Rmiss:')
        print(Rmiss_err_3dp)
        print('___________________________________')
        
    if replot==True:
  
        bin_centres=[];
        for i in range(len(h[1])-1):
            bin_centres.append((h[1][i]+h[1][i+1])/2)
  
        fig, ax= plt.subplots(1,figsize=(8,6),sharex=True)
        ax.bar(bin_centres,Rmiss_SM,width=bin_widths, yerr=err_Rmiss_SM,
               edgecolor='red', fill=False, capsize=4, ecolor='red', label=r'MG5(LO) SM prediction')
        ax.bar(bin_centres,Rmiss_sum, width=bin_widths,  yerr=err_Rmiss_quad,
               fill=False, edgecolor='blue', linestyle='--', capsize=4, ecolor='blue', label=r'MG5(LO) SM+BSM prediction' )
        ax.errorbar(bin_centres, meas, yerr=meas_stat, capsize=4, fmt='ok', label='Measured Data (& stat. error)')
        ax.set_ylabel(ylabel, fontsize=14)
        if plot=='a':
            ax.set_ylim(0,max(Rmiss_sum)*1.5+2.0)
        else:
            ax.set_ylim(0,max(Rmiss_sum)*1.5)
        ax.legend(loc="upper left", fontsize=12, facecolor='white', edgecolor='black')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_yscale('linear')
        ax.text(0.75, 0.93, '$\Lambda_{EFT}$ = '+str(lda_new)+' GeV', fontsize=12, transform=ax.transAxes)
        ax.text(0.75, 0.88,  '$M_{\chi}$   = '+str(m_chi)+' GeV', fontsize=12, transform=ax.transAxes)
        ax.text(0.75, 0.80,  r'$\frac{\sigma_{HD}}{\sigma_{MG}}=$'+str(np.round(rescale_factor,2))+'', fontsize=14, transform=ax.transAxes)
        
        plt.show()
        
        save_to='/Users/Genis/mphys/Rivet/DM/HNoE/plots/plots_nov_26'; 
        plt.savefig(save_to+'/Plot '+plot+'. rescaled Rmiss at m'+str(m_chi)+'l'+str(lda_new)+'.pdf')
        
    return [Rmiss_SM, Rmiss_DM, err_Rmiss_SM, err_Rmiss_DM, rescale_factor]
#______________________________________________________________________________
def get_cov(lda_new, scale_factors, save=True):    
    data_a=rescale_data(lda_new, scale_factors, 'a', False, True, False); plt.close()
    Rmiss_SM_a=data_a[0]; Rmiss_DM_a=data_a[1]; err_Rmiss_SM_a=data_a[2]; err_Rmiss_DM_a=data_a[3];
    
    data_b=rescale_data(lda_new, scale_factors, 'b', False, True, False); plt.close()
    Rmiss_SM_b=data_b[0]; Rmiss_DM_b=data_b[1]; err_Rmiss_SM_b=data_b[2]; err_Rmiss_DM_b=data_b[3];
    
    data_c=rescale_data(lda_new, scale_factors, 'c', False, True, False); plt.close()
    Rmiss_SM_c=data_c[0]; Rmiss_DM_c=data_c[1]; err_Rmiss_SM_c=data_c[2]; err_Rmiss_DM_c=data_c[3];
    
    data_d=rescale_data(lda_new, scale_factors, 'd', False, True, False); plt.close()
    Rmiss_SM_d=data_d[0]; Rmiss_DM_d=data_d[1]; err_Rmiss_SM_d=data_d[2]; err_Rmiss_DM_d=data_d[3];
    
    # Combine all into 1.
    Rmiss_SM=Rmiss_SM_a+Rmiss_SM_b+Rmiss_SM_c+Rmiss_SM_d
    Rmiss_DM=Rmiss_DM_a+Rmiss_DM_b+Rmiss_DM_c+Rmiss_DM_d
    err_Rmiss_SM = err_Rmiss_SM_a + err_Rmiss_SM_b + err_Rmiss_SM_c + err_Rmiss_SM_d   
    err_Rmiss_DM = err_Rmiss_DM_a + err_Rmiss_DM_b + err_Rmiss_DM_c + err_Rmiss_DM_d  
    
    cov_stat_SM=[x**2 for x in err_Rmiss_SM]
    cov_stat_DM=[x**2 for x in err_Rmiss_DM]
    
    cov_stat_SM=np.diag(cov_stat_SM)
    cov_stat_DM=np.diag(cov_stat_DM)
    
    print('Statistical covariance matrix for SM prediction:')
    print(np.around(cov_stat_SM, decimals=4))
    #print('Statistical covariance matrix for DM prediction:')
    #print(np.around(cov_stat_DM, decimals=5))
    
    
    if save==True:  	
        with open('MG_Data_m700.pickle', 'wb') as f:
            pickle.dump([m_chi, lda_new, Rmiss_SM, Rmiss_DM, cov_stat_SM, cov_stat_DM], f)
    

    return [Rmiss_SM, Rmiss_DM, cov_stat_SM]

#=============================================================================#
#                  Studying jpt1min generator level cuts:
#=============================================================================#
def small_range(sf):
    jpt1_20=[pandas.read_csv(path_2+"jpt20_nocuts.csv")]
    jpt1_50=[pandas.read_csv(path_2+"jpt1_50_nocuts.csv")]
    jpt1_60=[pandas.read_csv(path_2+"jpt1_60nocuts.csv")]
    jpt1_70=[pandas.read_csv(path_2+"jpt70_nocuts.csv")]
    jpt1_80=[pandas.read_csv(path_2+"jpt80_nocuts.csv")]
        
    jpt120=jpt1_20[0]; jpt150=jpt1_50[0]; jpt160=jpt1_60[0]; jpt170=jpt1_70[0]; jpt180=jpt1_80[0];
    
    jpt120=jpt120.query("{}{}{}".format(('jpt1'),"<",150))
    jpt150=jpt150.query("{}{}{}".format(('jpt1'),"<",150))
    jpt160=jpt160.query("{}{}{}".format(('jpt1'),"<",150))
    jpt170=jpt170.query("{}{}{}".format(('jpt1'),"<",150))
    jpt180=jpt180.query("{}{}{}".format(('jpt1'),"<",150))
    
    fig, ax = plt.subplots(1,figsize=(10,6),sharex=True)
    fig.suptitle('Number of events against leading jet $p_T$',fontsize=16)     
    ax.set_yscale('log')
    
    h=ax.hist([jpt120['jpt1'], jpt150['jpt1'], jpt160['jpt1'], jpt170['jpt1'],jpt180['jpt1']], 
    weights = [jpt120['weight']*sf[0], jpt150['weight']*sf[1], jpt160['weight']*sf[2], jpt170['weight']*sf[3], jpt180['weight']*sf[4]], 
    bins=70,color=['red','blue','green','black','magenta'], stacked=False,
    fill=False, histtype='step', label=['$p_T\ j_1$: 20','$p_T\ j_1$: 50','$p_T\ j_1$: 60', '$p_T\ j_1$: 70','$p_T\ j_1$: 80'])
    
    ax.set_ylabel('Number of events', fontsize=14)
    ax.set_xlabel('Leading jet $pT$ (GeV)',fontsize=14)
    plt.axvline(x=80,color='cyan')
    ax.set_xlim([25, 150])
    ax.legend(loc='lower right')
    save_to='/Users/Genis/mphys/Rivet/DM/HNoE/plots/plots_nov_26'; 
    plt.savefig(save_to+'/full_jpt1_plots.pdf')
    
    fig, ax2 = plt.subplots(1,figsize=(10,6),sharex=True)
    fig.suptitle('Number of events against leading jet $p_T$',fontsize=16)  
    h_2=ax2.hist([jpt120['jpt1'], jpt150['jpt1'], jpt160['jpt1'], jpt170['jpt1'],jpt180['jpt1']], 
    weights = [jpt120['weight']*sf[0], jpt150['weight']*sf[1], jpt160['weight']*sf[2], jpt170['weight']*sf[3], jpt180['weight']*sf[4]], 
    bins=70,color=['red','blue','green','black','magenta'], stacked=False,
    fill=False, histtype='step', label=['jpt1_20', 'jpt1_50','jpt1_60','jpt1_70','jpt1_80'])
    
    ax2.set_yscale('log')
    ax2.set_ylabel('Number of events', fontsize=14)
    ax2.set_xlabel('Leading jet $pT$ (GeV)',fontsize=14)
    plt.axvline(x=80,color='cyan')
    ax2.set_xlim([75, 85])
    ax2.set_ylim([5e5, 3e6 ])
    ax2.legend(loc='upper right')
    plt.show() 

    
    boi=[]; 
    for i in range(28,35):
        boi.append((h[1][i]))
    
    sigma=[];
    
    print('Jpt1 20')
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(0))
    sys.stdout.flush()
    
    for k in range (len(boi)-1):
        jpt120_temp=jpt120.query("{}{}{}".format(('jpt1'),">",boi[k]))
        jpt120_temp=jpt120.query("{}{}{}".format(('jpt1'),"<",boi[k+1]))
        sum_weight_square=0;
        for row in range(len(jpt120_temp)):  
            weight=jpt120_temp.iloc[row]['weight'];
            weight_sq=weight*weight;
            sum_weight_square+=weight_sq;
        err = math.sqrt(sum_weight_square)*sf[0]
        sigma.append(err)
        sum_weight_square=0;
        sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format((k+1)/(len(boi)-1)*100))
        sys.stdout.flush()
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100))
    sys.stdout.flush()
    print('')
    print('Done with jpt1_20')    

    print('Jpt1 50')
    sigma_50=[];
    for k in range (len(boi)-1):
        jpt150_temp=jpt150.query("{}{}{}".format(('jpt1'),">",boi[k]))
        jpt150_temp=jpt150.query("{}{}{}".format(('jpt1'),"<",boi[k+1]))
        sum_weight_square=0;
        for row in range(len(jpt150_temp)):  
            weight=jpt150_temp.iloc[row]['weight'];
            weight_sq=weight*weight;
            sum_weight_square+=weight_sq;
        err = math.sqrt(sum_weight_square)*sf[1]
        sigma_50.append(err)
        sum_weight_square=0;
        sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format((k+1)/(len(boi)-1)*100))
        sys.stdout.flush()
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100))
    sys.stdout.flush()
    print('')
    print('Done with jpt1_50')    
    
    print('Jpt1 60')
    sigma_60=[];
    for k in range (len(boi)-1):
        jpt160_temp=jpt160.query("{}{}{}".format(('jpt1'),">",boi[k]))
        jpt160_temp=jpt160.query("{}{}{}".format(('jpt1'),"<",boi[k+1]))
        sum_weight_square=0;
        for row in range(len(jpt160_temp)):  
            weight=jpt160_temp.iloc[row]['weight'];
            weight_sq=weight*weight;
            sum_weight_square+=weight_sq;
        err = math.sqrt(sum_weight_square)*sf[2]
        sigma_60.append(err)
        sum_weight_square=0;
        sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format((k+1)/(len(boi)-1)*100))
        sys.stdout.flush()
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100))
    sys.stdout.flush()
    print('')
    print('Done with jpt1_60')   
    
    print('Jpt1 70')
    sigma_70=[];
    for k in range (len(boi)-1):
        jpt170_temp=jpt170.query("{}{}{}".format(('jpt1'),">",boi[k]))
        jpt170_temp=jpt170.query("{}{}{}".format(('jpt1'),"<",boi[k+1]))
        sum_weight_square=0;
        for row in range(len(jpt170_temp)):  
            weight=jpt170_temp.iloc[row]['weight'];
            weight_sq=weight*weight;
            sum_weight_square+=weight_sq;
        err = math.sqrt(sum_weight_square)*sf[3]
        sigma_70.append(err)
        sum_weight_square=0;
        sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format((k+1)/(len(boi)-1)*100))
        sys.stdout.flush()
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100))
    sys.stdout.flush()
    print('')
    print('Done with jpt1_70')   

    print('Jpt1 80')
    sigma_80=[];
    for k in range (len(boi)-1):
        jpt180_temp=jpt180.query("{}{}{}".format(('jpt1'),">",boi[k]))
        jpt180_temp=jpt180.query("{}{}{}".format(('jpt1'),"<",boi[k+1]))
        sum_weight_square=0;
        for row in range(len(jpt180_temp)):  
            weight=jpt180_temp.iloc[row]['weight'];
            weight_sq=weight*weight;
            sum_weight_square+=weight_sq;
        err = math.sqrt(sum_weight_square)*sf[4]
        sigma_80.append(err)
        sum_weight_square=0;
        sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format((k+1)/(len(boi)-1)*100))
        sys.stdout.flush()
    sys.stdout.write("\rCalculating errors, please wait ...  {:.0f}%".format(100))
    sys.stdout.flush()
    print('')
    print('Done with jpt1_80')   

    bin_centres=[];
    for i in range(len(boi)-1):
        bin_centres.append((boi[i]+boi[i+1])/2)
    
    boi_peaks=[]; boi_peaks_50=[]; boi_peaks_60=[]; boi_peaks_70=[]; boi_peaks_80=[];
    for i in range(28, 34):
        boi_peaks.append(h[0][0][i])
        boi_peaks_50.append(h[0][1][i])
        boi_peaks_60.append(h[0][2][i])
        boi_peaks_70.append(h[0][3][i])
        boi_peaks_80.append(h[0][4][i])
    
    fig, ax3= plt.subplots(1,figsize=(10,6),sharex=True)
    fig.suptitle('Number of events against leading jet $p_T$',fontsize=16)  
    ax3.bar(bin_centres,boi_peaks, width=h[1][1]-h[1][0], yerr=sigma, 
            edgecolor='red', fill=False, capsize=4, ecolor='red', label='$p_T\ j_1$: 20')
    ax3.bar(bin_centres,boi_peaks_50, width=h[1][1]-h[1][0], yerr=sigma_50, 
            edgecolor='blue', fill=False, capsize=4, ecolor='blue', label='$p_T\ j_1$: 50')
    ax3.bar(bin_centres,boi_peaks_60, width=h[1][1]-h[1][0], yerr=sigma_60, 
            edgecolor='green', fill=False, capsize=4, ecolor='green', label='$p_T\ j_1$: 60')
    ax3.bar(bin_centres,boi_peaks_70, width=h[1][1]-h[1][0], yerr=sigma_70, 
            edgecolor='black', fill=False, capsize=4, ecolor='black', label='$p_T\ j_1$: 70')
    ax3.bar(bin_centres,boi_peaks_80, width=h[1][1]-h[1][0], yerr=sigma_80, 
            edgecolor='magenta', fill=False, capsize=4, ecolor='magenta', label='$p_T\ j_1$: 80')
    
        
    ax3.set_yscale('log')
    ax3.set_ylabel('Number of events', fontsize=14)
    ax3.set_xlabel('Leading jet $pT$', fontsize=14)
    plt.axvline(x=80,color='cyan')
    ax3.set_xlim([75, 85 ]) 
    ax3.set_ylim([5e5, 5e6 ])   
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(loc='upper right')
    plt.savefig(save_to+'/zoomed_jpt1_plots.pdf')
    
    
    return h

#______________________________________________________________________________
# Plot ratio to pjt1_min = 20 GeV_
def ratio_range(sf):
    jpt1_20=[pandas.read_csv(path_2+"jpt20_nocuts.csv")]
    jpt1_30=[pandas.read_csv(path_2+"jpt30_nocuts.csv")]
    jpt1_40=[pandas.read_csv(path_2+"jpt40_nocuts.csv")]
    jpt1_50=[pandas.read_csv(path_2+"jpt1_50_nocuts.csv")]
    jpt1_60=[pandas.read_csv(path_2+"jpt1_60nocuts.csv")]
    jpt1_70=[pandas.read_csv(path_2+"jpt70_nocuts.csv")]
    jpt1_80=[pandas.read_csv(path_2+"jpt80_nocuts.csv")]
        
    jpt120=jpt1_20[0]; jpt130=jpt1_30[0]; jpt140=jpt1_40[0]; jpt150=jpt1_50[0]; jpt160=jpt1_60[0]; jpt170=jpt1_70[0]; jpt180=jpt1_80[0];
    
    jpt120=jpt120.query("{}{}{}".format(('jpt1'),"<",150))
    jpt130=jpt130.query("{}{}{}".format(('jpt1'),"<",150))
    jpt140=jpt140.query("{}{}{}".format(('jpt1'),"<",150))
    jpt150=jpt150.query("{}{}{}".format(('jpt1'),"<",150))
    jpt160=jpt160.query("{}{}{}".format(('jpt1'),"<",150))
    jpt170=jpt170.query("{}{}{}".format(('jpt1'),"<",150))
    jpt180=jpt180.query("{}{}{}".format(('jpt1'),"<",150)) 
    
    fig, ax = plt.subplots(1,figsize=(10,6),sharex=True)
    fig.suptitle('Number of events against pt of leading jet',fontsize=16)     
    ax.set_yscale('log')
    ax.set_facecolor('#d3d3d3')
    
    h=ax.hist([jpt120['jpt1'], jpt130['jpt1'], jpt140['jpt1'], jpt150['jpt1'], jpt160['jpt1'], jpt170['jpt1'],jpt180['jpt1']], 
    weights = [jpt120['weight']*sf[0], jpt130['weight']*sf[1], jpt140['weight']*sf[2], jpt150['weight']*sf[3], jpt160['weight']*sf[4], jpt170['weight']*sf[5], jpt180['weight']*sf[6]], 
    bins=70,color=['red','magenta','orange','blue','green','black','yellow'], stacked=False,
    fill=False, histtype='step', label=['jpt1_20','jpt1_30','jpt1_40','jpt1_50','jpt1_60','jpt1_70','jpt1_80'])
    
    ax.set_ylabel('Number of events', fontsize=14)
    ax.set_xlabel('pT of leading jet',fontsize=14)
    plt.axvline(x=80,color='cyan')
    ax.set_xlim([25, 150])
    
    
    ax.legend(loc="lower right", fontsize=11)
    
    jpt120_peaks=h[0][0]; jpt130_peaks=h[0][1]; jpt140_peaks=h[0][2];
    jpt150_peaks=h[0][3]; jpt160_peaks=h[0][4]; jpt170_peaks=h[0][5];
    jpt180_peaks=h[0][6];
    
    fig, ax2 = plt.subplots(1,figsize=(10,6),sharex=True)
    fig.suptitle('$pTj^1$ plots relative to $pTj^1$: 20 GeV',fontsize=16)     
    ax2.set_yscale('linear')
    ax2.set_facecolor('#d3d3d3')
    
    bin_centres=[];
    for i in range(len(h[1])-1):
        bin_centres.append((h[1][i]+h[1][i+1])/2)                  
                      
    ax2.step(bin_centres,jpt120_peaks/jpt120_peaks, 'r--', label='N: 20', linewidth=2)
    ax2.step(bin_centres,jpt130_peaks/jpt120_peaks,  'm-', label='N: 30', linewidth=1)
    ax2.step(bin_centres,jpt140_peaks/jpt120_peaks, 'C1',  label='N: 40', linewidth=1)
    ax2.step(bin_centres,jpt150_peaks/jpt120_peaks, 'b-',  label='N: 50', linewidth=1)
    ax2.step(bin_centres,jpt160_peaks/jpt120_peaks, 'g-',  label='N: 60', linewidth=1)
    ax2.step(bin_centres,jpt170_peaks/jpt120_peaks, 'k-',  label='N: 70', linewidth=1)
    ax2.step(bin_centres,jpt180_peaks/jpt120_peaks, 'C8',  label='N: 80', linewidth=1)
    ax2.set_ylabel(r'$\frac{pTj^1: \ N}{pTj^1: \ 20}$', fontsize=14)
    ax2.set_xlabel('pT of leading jet',fontsize=14)
    ax2.set_ylim(0.1,1.75)
    ax2.set_xlim(25,150)
    ax2.legend(loc="upper left", fontsize=11)