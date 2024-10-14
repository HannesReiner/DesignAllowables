import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import variation
from scipy.stats import norm 
from scipy.stats import f 
from scipy.stats import alpha
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
from SALib.analyze import hdmr
import pickle

# Load MAT219 input parameters (see Table 2)
with open("Xcheck","rb") as file_handle:
    X = pickle.load(file_handle)
# Load 5% post-peak forces from FEA simulation 
with open("Yafter","rb") as file_handle:
    Y_after = pickle.load(file_handle)

# Define variables of HDMR and choose ranges (according to FEA input parameters)           
problem = {
    'num_vars': 8,
    'names': ['Emod1', 'Emod2', 'G12','nu12','epsfi','epsmi','epsfs','epsms'],
    'bounds': [[140000,175000],
               [7000,11000],
               [2000,8000],
               [0.05,0.49],
               [0.1,1.0],
               [0.05,1.0],
               [5.0,50.0],
               [1.0,10.0]],
}

# Analyse first- and second-order sensitivity indices
H1_after = hdmr.analyze(problem, X, Y_after, 2, 100, 5)

# Plot results (see Figure 7)
patterns = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
plt.rcParams.update({'font.size': 18})
fig = plt.subplots(figsize =(12, 8))
ax = plt.axes()
#fig, ax1 = plt.subplots(figsize =(16, 9))
plt.bar([1,2,3,4,6,8], [H1_after['Sa'][0], H1_after['Sa'][1], H1_after['Sa'][2], H1_after['Sa'][3], H1_after['Sa'][5], H1_after['Sa'][7]], color='gray', label='insensitive FE inputs', hatch=patterns[0])
plt.bar([5,7], [H1_after['Sa'][4], H1_after['Sa'][6]], color='palegreen', label='sensitive first-oder FE inputs', hatch=patterns[6])
plt.bar([12], [H1_after['Sa'][31]], color='red', label='higher-order correlation', hatch=patterns[9])
plt.ylabel('Sensitivity index')
plt.xlabel('FE input parameter')
#ax.set_xlabel('FE input parameter', loc = 'left') 
plt.legend(loc="upper left",ncol = 1)
plt.grid()
plt.ylim(0, 1)
ax.set_xticks([1,2,3,4,5,6,7,8, 12])
ax.set_xticklabels(['$E_1$', '$E_2$','$G_{12}$','$\\nu_{12}$', '$\epsilon_1^i$', '$\epsilon_2^i$', '$\epsilon_1^s$', '$\epsilon_2^s$', '$\epsilon_1^i$ - $\epsilon_1^s$'])
plt.text(1-0.45, H1_after['Sa'][0]+0.02,str(round((ax.patches[0].get_height()), 2)),fontsize = 18,color ='black')
plt.text(2-0.3, H1_after['Sa'][1]+0.02,str(round((ax.patches[1].get_height()), 2)),fontsize = 18,color ='black')
plt.text(3-0.3, H1_after['Sa'][2]+0.02,str(round((ax.patches[2].get_height()), 2)),fontsize = 18,color ='black')
plt.text(4-0.3, H1_after['Sa'][3]+0.02,str(round((ax.patches[3].get_height()), 2)),fontsize = 18,color ='black')
plt.text(5-0.45, H1_after['Sa'][4]+0.02,str(round((ax.patches[6].get_height()), 2)),fontsize = 18,color ='black')
plt.text(6-0.3, H1_after['Sa'][5]+0.02,str(round((ax.patches[4].get_height()), 2)),fontsize = 18,color ='black')
plt.text(7-0.45, H1_after['Sa'][6]+0.02,str(round((ax.patches[7].get_height()), 2)),fontsize = 18,color ='black')
plt.text(8-0.3, H1_after['Sa'][7]+0.02,str(round((ax.patches[5].get_height()), 2)),fontsize = 18,color ='black')
plt.text(12-0.45, H1_after['Sa'][31]+0.02,str(round((ax.patches[8].get_height()), 2)),fontsize = 18,color ='black')
plt.savefig('Sensitivity_5PerCent.png', bbox_inches='tight')
plt.show()

