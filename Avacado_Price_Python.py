# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:47:56 2021

@author: UMANG
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:12:47 2021

@author: UMANG
"""

# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
ap= pd.read_csv(r"C:\Users\UMANG\OneDrive\Desktop\MLR2/Avacado_Price.csv")


ap = ap.drop(ap.iloc[:, 8:13], axis = 1)
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, eap.)

ap.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#Tot_ava3
plt.bar(height = ap.tot_ava3, x = np.arange(1, 82, 1))
plt.hist(ap.tot_ava3) #histogram
plt.boxplot(ap.tot_ava3) #boxplot

# AveragePrice
plt.bar(height = ap.AveragePrice, x = np.arange(1, 82, 1))
plt.hist(ap.AveragePrice) #histogram
plt.boxplot(ap.AveragePrice) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=ap['tot_ava3'], y=ap['AveragePrice'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(ap['tot_ava3'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(ap.AveragePrice, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(ap.iloc[:, :])
                             
# Correlation matrix 
ap.corr()

# we see there exists High collinearity between input variables especially between


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags' , data = ap).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

ap_new = ap.drop(ap.index[[80]])

# Preparing model                  
ml_new = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Total_Bags + Small_Bags + Large_Bags', data = ap_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
#

#
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Large_Bags', data = ap).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(ap)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = ap.AveragePrice, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
ap_train, ap_test = train_test_split(ap, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('AveragePrice ~ Total_Volume + tot_ava1 + tot_ava2 + tot_ava3 + Small_Bags + Large_Bags', data = ap_train).fit()

# prediction on test data set 
test_pred = model_train.predict(ap_test)

# test residual values 
test_resid = test_pred - ap_test.MPG
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(ap_train)

# train residual values 
train_resid  = train_pred - ap_train.MPG
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
