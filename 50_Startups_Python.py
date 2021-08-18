# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:32:23 2021

@author: UMANG
"""

# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
sp = pd.read_csv(r"C:\Users\UMANG\OneDrive\Desktop\MLR2/50_Startups.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

sp.describe()
sp = sp.drop(["state"],axis = 1)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# spend
plt.bar(height = sp.spend, x = np.arange(1, 82, 1))
plt.hist(sp.spend) #histogram
plt.boxplot(sp.spend) #boxplot

# adm
plt.bar(height = sp.adm, x = np.arange(1, 82, 1))
plt.hist(sp.adm) #histogram
plt.boxplot(sp.adm) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=sp['spend'], y=sp['profit'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(sp['spend'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(sp.profit, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(sp.iloc[:, :])
                             
# Correlation matrix 
sp.corr()

# we see there exists High collinearity between input variables especially between


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('profit ~ spend + adm + mspend', data = sp).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 45and49 is showing high influence so we can exclude that entire row

sp_new = sp.drop(sp.index[[45,49]])

# Preparing model                  
ml_new = smf.ols('profit ~ spend + adm + mspend', data = sp_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('spend ~ adm + mspend', data = sp).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_wt = smf.ols('adm ~ spend + mspend', data = sp).fit().rsquared  
vif_wt = 1/(1 - rsq_wt)

rsq_vol = smf.ols('mspend ~ adm + spend', data = sp).fit().rsquared  
vif_vol = 1/(1 - rsq_vol) 

# Storing vif values in a data frame
d1 = {'Variables':['spend', 'adm', 'mspend'], 'VIF':[vif_hp, vif_wt, vif_vol]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('profit ~  adm + mspend', data = sp).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(sp)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = sp.profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
sp_train, sp_test = train_test_split(sp, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('profit ~  adm + mspend', data = sp_train).fit()

# prediction on test data set 
test_pred = model_train.predict(sp_test)

# test residual values 
test_resid = test_pred - sp_test.profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(sp_train)

# train residual values 
train_resid  = train_pred - sp_train.profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
