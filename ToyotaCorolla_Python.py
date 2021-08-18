# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:12:47 2021

@author: UMANG
"""

# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
tc = pd.read_csv(r"C:\Users\UMANG\OneDrive\Desktop\MLR2/ToyotaCorolla.csv",encoding= 'unicode_escape')

tc = tc.drop(["Id","Model","Mfg_Month","Mfg_Year","Fuel_Type","Met_Color","Color","Automatic","Cylinders","Mfr_Guarantee",],axis = 1)
tc = tc.drop(tc.iloc[:, 9:42], axis = 1)
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

tc.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#HP
plt.bar(height = tc.HP, x = np.arange(1, 82, 1))
plt.hist(tc.HP) #histogram
plt.boxplot(tc.HP) #boxplot

# Price
plt.bar(height = tc.Price, x = np.arange(1, 82, 1))
plt.hist(tc.Price) #histogram
plt.boxplot(tc.Price) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=tc['HP'], y=tc['Price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(tc['HP'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(tc.Price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(tc.iloc[:, :])
                             
# Correlation matrix 
tc.corr()

# we see there exists High collinearity between input variables especially between


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04 + KM + cc + HP + Doors + Gears + Quarterly_Tax + Weight ' , data = tc).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

tc_new = tc.drop(tc.index[[80]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + cc + HP + Doors + Gears + Quarterly_Tax + Weight', data = tc_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
#

#
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Price ~ Age_08_04 + KM + cc + HP + Gears + Quarterly_Tax + Weight', data = tc).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(tc)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = tc.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
tc_train, tc_test = train_test_split(tc, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04 + KM + cc + HP + Gears + Quarterly_Tax + Weight', data = tc_train).fit()

# prediction on test data set 
test_pred = model_train.predict(tc_test)

# test residual values 
test_resid = test_pred - tc_test.MPG
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(tc_train)

# train residual values 
train_resid  = train_pred - tc_train.MPG
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
