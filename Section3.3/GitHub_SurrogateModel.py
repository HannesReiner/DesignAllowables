import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import variation
from scipy.stats import norm 
from scipy.stats import f 
from scipy.stats import alpha
import seaborn as sns
from bisect import bisect_left
from SALib.analyze import hdmr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle


# Load MAT219 input parameters (see Table 2)
with open("Xcheck","rb") as file_handle:
    X = pickle.load(file_handle)
# Load 5% ppost-peak forces from FEA simulation 
with open("Yafter","rb") as file_handle:
    Y_after = pickle.load(file_handle)

# Create vector of the two sensitive FEA input parameters for damage initiation and damage saturation in fibre direction
eps_fi_fs_vect = []
for ki in X:
    eps_fi_fs_vect.append([ki[4], ki[6]])

# Normalise FEA input parameters (range from 0 to 1) and define training and testing data
scaler_eps_fi_fs_after = MinMaxScaler()
eps_fi_fs_train_after, eps_fi_fs_test_after, Y_eps_fi_fs_train_after, Y_eps_fi_fs_test_after = train_test_split(scaler_eps_fi_fs_after.fit_transform(np.array(eps_fi_fs_vect).reshape(-1, 2)), np.array(Y_after).reshape(1, -1).transpose(), test_size=0.25, random_state=42)


# Create second-order polynomial from FEA input parameters for training
x_after = PolynomialFeatures(degree=2, include_bias=False).fit_transform(eps_fi_fs_train_after)
# Run polynomial regression
model_after = LinearRegression().fit(x_after, Y_eps_fi_fs_train_after)
# Training accuracy
print('Training accuracy:', model_after.score(x_after, Y_eps_fi_fs_train_after))
# Beta coefficients (see Table 4)
print(f"beta_0: {model_after.intercept_}")
print(f"beta_1 to beta_5: {model_after.coef_}")

# Save beta coefficients
with open("PolynRegression_intercept_after", "wb") as fp:   #Pickling
    pickle.dump(model_after.intercept_, fp)
with open("PolynRegression_coef_after", "wb") as fp:   #Pickling
    pickle.dump(model_after.coef_, fp)

# Testing of surrogate model
x_test = PolynomialFeatures(degree=2, include_bias=False).fit_transform(eps_fi_fs_test_after)
model_after.predict(x_test)
# Testing accuracy (see Figure 8)
print('Testing accuracy:', r2_score(Y_eps_fi_fs_test_after, model_after.predict(x_test)))

# Scale training dnd testing inputs back to original ranges
eps_fi_fs_train_scaled = scaler_eps_fi_fs_after.inverse_transform(np.array(eps_fi_fs_train_after).reshape(-1, 2))
eps_fi_fs_test_scaled = scaler_eps_fi_fs_after.inverse_transform(np.array(eps_fi_fs_test_after).reshape(-1, 2))
