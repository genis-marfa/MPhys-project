import Functions_library as fl
import matplotlib.pyplot as plt
import numpy as np
import pickle

luminosity = 140.*1000 #in pb = 300 fb^-1

sum_of_weights=[29715.6,2354.22,6.33457];
cross_sections=[4.28E+04,2.26E+03,6.334571];
scale_factors=[luminosity * xs / sw for xs,sw in zip(cross_sections,sum_of_weights)];

samples=["vbf","mono"]; 


#-------------------------------------------------------------------------------------------#
# Studying pt for leading jet:
samples=['mono','vbf']
sow=[2344.45, 584.319,  379.663, 265.881, 189.773];
cs=[2.25E+03, 5.568428e+02, 3.796626e+02, 265.8811, 189.773];
sf=[luminosity * xs / sw for xs,sw in zip(cs,sow)];

# fl.small_range(sf)
# fl.ratio_range(sf)


#-------------------------------------------------------------------------------------------#    

scale_factors[:]=[x/(140.*1000) for x in scale_factors]
fl.rescale_data(400, scale_factors, 'd')
#fl.histoplot(l, scale_factors, plot, show)
# CREATES HISTOGRAM (no errors) FOR CERTAIN PLOT AT CERTAIN LAMBDA:
# l: Specify lambda
# Scale factors: Defined as cross-section / sum of weights.
# plot: 'a','b','c' or 'd': Plots corresponding diagram to that of pg 16 figure 4.
#   - 'a': cross-section vs met for > 1 jet region.
#   - 'b': cross-section vs met for VBF region.
#   - 'c': cross-section vs mjj for VBF region.
#   - 'd': cross-section vs dphi for VBF region.
# show: Boolean: Show histogram (set to true by default). 

#h=fl.histoplot(100, scale_factors, 'b')

# fl.errorbar(plot, scale_factors, l_i, l_f, l++, show, summ):
# CREATES A PLOT WITH ERRORBARS:
# plot: 'a','b','c' or 'd': Plots corresponding diagram to that of pg 16 figure 4.
#   - 'a': cross-section vs met for > 1 jet region.
#   - 'b': cross-section vs met for VBF region.
#   - 'c': cross-section vs mjj for VBF region.
#   - 'd': cross-section vs dphi for VBF region.
# Scale factors: Defined as cross-section / sum of weights.
# l_i,l_f, l++: Makes a 1D scan in lambda. Starts at initial lambda l_i, runs
# up to final lambda l_f, in the specified increment l++.
# show: Boolean: Show temporary histogram (no errorbars) (set to true by default).
# summ: Boolean: Show summary at the end of error calculations (set to true by default).

#fl.errorbar_plot('a', scale_factors, 100, 100, 50, False)

#fl.hepdata_compare(plot,scale_factors, False, False)
# COMPARES DATA TO HEPDATA TABLES (1-4)
# plot: 'a','b','c' or 'd': Plots corresponding diagram to that of pg 16 figure 4.
#   - 'a': cross-section vs met for > 1 jet region. (HepData Table 1).
#   - 'b': cross-section vs met for VBF region. (HepData Table 2).
#   - 'c': cross-section vs mjj for VBF region. (HepData Table 3).
#   - 'd': cross-section vs dphi for VBF region. (HepData Table 4).
# Scale factors: Defined as cross-section / sum of weights.
# show: Boolean: Show temporary histogram (no errorbars) (set to true by default).
# summ: Boolean: Show summary at the end of error calculations (set to true by default).

#plots=['a','b','c','d']
#for plot in plots:
#    fl.hepdata_compare(plot, scale_factors, show=False, summ=True)

lda=400;
print('LAMBDA:', lda)

#fl.rescale_data(lda, scale_factors, plot, replot=True)
#fl.get_cov(lda,scale_factors, False)

