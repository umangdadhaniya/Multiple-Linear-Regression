# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:48:05 2021

@author: UMANG
"""

# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
cod = pd.read_csv(r"C:\Users\UMANG\OneDrive\Desktop\MLR2/Computer_Data.csv")
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

cod.describe()
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
cod['cd']= label_encoder.fit_transform(cod['cd'])
cod['multi']= label_encoder.fit_transform(cod['multi'])
cod['premium']= label_encoder.fit_transform(cod['premium'])

cod = cod.drop(["X"],axis = 1)  

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# sped
plt.bar(height = cod.speed, x = np.arange(1, 82, 1))
plt.hist(cod.speed) #histogram
plt.boxplot(cod.speed) #boxplot

# MPG
plt.bar(height = cod.price, x = np.arange(1, 82, 1))
plt.hist(cod.price) #histogram
plt.boxplot(cod.price) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cod['speed'], y=cod['price'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cod['speed'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cod.price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cod.iloc[:, :])
                             
# Correlation matrix 
cod.corr()

# we see there exists High collinearity between input variables especially between


# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = cod).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals


cod_new = cod.drop(cod.index[[1440,1700]])

# Preparing model                  
ml_new = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = cod_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed ~ hd + ram + screen + cd + multi + premium + ads + trend', data = cod).fit().rsquared  
vif_speed = 1/(1 - rsq_speed) 

rsq_hd = smf.ols('hd ~ speed + ram + screen + cd + multi + premium + ads + trend', data = cod).fit().rsquared  
vif_hd = 1/(1 - rsq_hd) 

rsq_ram = smf.ols('ram ~ hd + speed + screen + cd + multi + premium + ads + trend', data = cod).fit().rsquared  
vif_ram = 1/(1 - rsq_ram)  

rsq_screen = smf.ols('screen ~ hd + ram + speed + cd + multi + premium + ads + trend', data = cod).fit().rsquared  
vif_screen = 1/(1 - rsq_screen) 

rsq_cd = smf.ols('cd ~ hd + ram + screen + speed + multi + premium + ads + trend', data = cod).fit().rsquared  
vif_cd = 1/(1 - rsq_cd) 

rsq_multi = smf.ols('multi ~ hd + ram + screen + cd + speed + premium + ads + trend', data = cod).fit().rsquared  
vif_multi = 1/(1 - rsq_multi) 

rsq_premium = smf.ols('premium ~ hd + ram + screen + cd + multi + speed + ads + trend', data = cod).fit().rsquared  
vif_premium = 1/(1 - rsq_premium) 

rsq_ads = smf.ols('ads ~ hd + ram + screen + cd + multi + premium + speed + trend', data = cod).fit().rsquared  
vif_ads = 1/(1 - rsq_ads) 

rsq_trend = smf.ols('trend ~ hd + ram + screen + cd + multi + premium + ads + speed', data = cod).fit().rsquared  
vif_trend = 1/(1 - rsq_trend) 


# Storing vif values in a data frame
d1 = {'Variables':['speed', 'hd', 'ram', 'screen','cd','multi','premium', 'ads','trend'], 'VIF':[vif_speed, vif_hd, vif_ram, vif_screen,vif_cd,vif_multi,vif_premium,vif_ads,vif_trend,]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price ~ speed + ram + screen + cd + multi + premium + ads + trend', data = cod).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cod)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cod.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cod_train, cod_test = train_test_split(cod, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ speed + ram + screen + cd + multi + premium + ads + trend', data = cod_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cod_test)

# test residual values 
test_resid = test_pred - cod_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cod_train)

# train residual values 
train_resid  = train_pred - cod_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
