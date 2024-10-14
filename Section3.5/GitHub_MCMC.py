import os
import h5py
import numpy as np
from tensorflow.python import keras
import tensorflow as tf
import math
from scipy.optimize import minimize
from scipy.stats import norm
import emcee
import arviz as ar
import time
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

global coef

# Load beta coefficients for surrogate model
import pickle
with open("PolynRegression_intercept_after","rb") as file_handle:
    poly_intercept = pickle.load(file_handle)
with open("PolynRegression_coef_after","rb") as file_handle:
    poly_coef = pickle.load(file_handle)

# Beta coefficients for surrogate model
coef = np.append(poly_intercept, poly_coef[0])

# Define the surrogate model (from polynomial regression, see Equation 14)
def forward_model(theta, coef):
    # Note theta = [e_i, e_s]
    poly = coef[0] + theta[0]*coef[1] + theta[1]*coef[2] + ((theta[0]**2)*coef[3]) + theta[0]*theta[1]*coef[4] + ((theta[1]**2)*coef[5])
    # Run the forawrd model
    return poly


# DEFINE LOG PRIOR (normal prior)
 def log_prior(theta, e_i_mu=0.6, e_i_sigma=.03,e_s_lower=0.0, e_s_upper=0.7):
     e_i = theta[0]
     e_s = theta[1]
     # are the uniform distr params in their ranges?
     if e_s_lower <= e_s <= e_s_upper:
         #return -1 # yes? return pdf of E
         return -np.log(e_i_sigma * np.sqrt(2*math.pi)) - 0.5*((e_i - e_i_mu)/e_i_sigma)**2
     else:
         return -np.inf # no? return -Inf = log(0)
    
# DEFINE LOG PRIOR (uniform prior)
#def log_prior(theta, e_i_lower=0.0, 
#              e_i_upper=1.0, e_s_lower=0.0, e_s_upper=0.7):
#    e_i = theta[0]
#    e_s = theta[1]
#    # are the uniform distr params in their ranges?
#    if e_i_lower <= e_i <= e_i_upper and e_s_lower <= e_s <= e_s_upper:
#        return 0 # yes? return pdf of E
#    else:
#        return -np.inf # no? return -Inf = log(0)


# function for the log_likelihood (see Equation 16)
def log_likelihood(theta, coef, mu_y=7809.6, sigma_y=552.5):
    S_OHT_hat = forward_model(theta, coef)
    return (-np.log(sigma_y * np.sqrt(2*math.pi)) - 0.5*((S_OHT_hat - mu_y)/sigma_y)**2)

# DEFINE LOG POSTERIOR
# function for the log-posterior
def log_post(theta):
    lprior = log_prior(theta)
    if not np.isfinite(lprior):
        return -np.inf
    else:
        return lprior + log_likelihood(theta, coef)

# PERFORM MLE TO INITIALIZE MCMC CHAINS
nll = lambda theta: -log_likelihood(theta, coef) # lambda function for negative loglikelihood

# Initialisation for optimisation - choose random points in the prior
e_i_init = np.random.uniform(0.01, 0.99,1)
e_s_init = np.random.uniform(0.01, 0.70,1)
initial = np.array([e_i_init, e_s_init])
initial = np.reshape(initial, (1,2))

# Define upper and lower bounds for FEA input parameters (normalised to range between 0 and 1)
bounds = ((0.01, 0.99), (0.01, 0.70))
soln = minimize(nll, initial[0])

print("The MLE is: ", soln.x)
print("The predicted 5% post-peak force at the MLE point is: ", forward_model(soln.x, coef))

# MCMC SAMPLING
nwalkers = 200 # number of walkers in the ensemble
ndim = 2 # nmumber of free variables

max_n = 20000 # maximum number of iterations
check = 100 # check convergence every check samples
auto_corr_multipier = 100 # chain length is greater than tau times this
delta_auto_corr = 0.01 # upper lim on percent change of tau

# inital points
initial_pnts = soln.x + 1e-4*np.random.randn(nwalkers, ndim)

# constuct emcee sampler object
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post)

# Run emcee with automatic checks for convergence
# Following code modified from https://emcee.readthedocs.io/en/stable/tutorials/monitor/
# Tracking of changes in average autocorrelation time estimate
index = 0
autocorr = np.empty(max_n)

# For testing convergence
old_tau = np.inf

# Sampling for up to max_n steps
start = time.time()
for sample in sampler.sample(initial_pnts, iterations=max_n, progress=True):
    # Only check convergence every check steps
    if sampler.iteration % check:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    # check if number of iterations is greater than the autocorrelation time
    converged = np.all(tau * auto_corr_multipier < sampler.iteration)
    # check if autocorrelation time is smaller than 
    converged &= np.all(np.abs(old_tau - tau) / tau < delta_auto_corr)
    if converged:
        break
    old_tau = tau
end = time.time()

elapsed = end - start 
print('total time', elapsed)

with open('timing_before.txt', 'w') as f:
    f.write('start time'+str(start))
    f.write('end time '+str(end))
    f.write('total time '+str(elapsed))


# Save the data to an arviz file
var_names = ["e_i", "e_s"]
idata1 = ar.from_emcee(sampler, var_names=var_names)
idata1.to_netcdf('PosteriorSamples')

idata1.sel(draw=slice(100, None))

# Plot marginal posterior distributions
fig = plt.subplots(figsize =(12, 8))
plt.rcParams.update({'font.size': 24})
_, bins, _ = plt.hist(idata1.posterior['e_i'][-1][-2000:], bins=np.arange(0,1,0.02), density=1,alpha=0.3, color = "royalblue",label='e_i')
_, bins, _ = plt.hist(idata1.posterior['e_s'][-1][-2000:], bins=np.arange(0,1,0.02), density=1,alpha=0.3, color = "limegreen",label='e_s')
plt.legend(loc="upper right")
plt.savefig('Posterior_distributions.png', bbox_inches='tight')
plt.show()


# Save posterior distributions
import pickle
with open("Init_post", "wb") as fp:   #Pickling
    pickle.dump(idata1.posterior.data_vars['e_i'].values, fp)


with open("Sat_post", "wb") as fp:   #Pickling
    pickle.dump(idata1.posterior.data_vars['e_s'].values, fp)
