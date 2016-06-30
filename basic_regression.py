# Basic linear regression model
# week2 assignment for Regression modeling in practice

# moderating variable: income per person
# explanator variable: alcohol consumption
# response variable: life expectancy
"""
Created on Wed Jun 29 17:33:19 2016

@author: taehee jeong
"""
# import libraries

import numpy as np
import pandas as pd
import statsmodels.api
import statsmodels.formula.api as smf
import seaborn
import matplotlib.pyplot as plt

# bug fix for display formats to avoid run time errors
#pandas.set_option('display.float_format', lambda x:'%.2f'%x)

#%% Load data
path='C:/Bigdata/Data Analysis and Interpretation/Dataset/GapMinder/'
data = pd.read_csv(path+'gapminder.csv', low_memory=False)

print data.columns

#setting variables you will be working with to numeric
#data['incomeperperson'] = pd.to_numeric(data['incomeperperson'], errors='coerce')
data['alcconsumption'] = pd.to_numeric(data['alcconsumption'], errors='coerce')
data['lifeexpectancy'] = pd.to_numeric(data['lifeexpectancy'], errors='coerce')


#%% centering

# subset of data
features=['alcconsumption','lifeexpectancy']
sub1=data[features]

# remove row with NA
sub1_clean=sub1.dropna()

#sub1_clean.alcconsumption.mean()
#sub1_clean.lifeexpectancy.mean()

alcoh_mean=sub1_clean['alcconsumption'].mean()
life_mean=sub1_clean['lifeexpectancy'].mean()


sub2=sub1_clean.copy()

sub2['alcconsumption']=sub2['alcconsumption'].apply(lambda x:x-alcoh_mean)
sub2['lifeexpectancy']=sub2['lifeexpectancy'].apply(lambda x:x-life_mean)

#sub2.alcconsumption.mean()
#sub2.lifeexpectancy.mean()

#%% basic linear regression

scat1 = seaborn.regplot(x="alcconsumption", y="lifeexpectancy", scatter=True, data=sub1)
plt.xlabel('alcohol consumption (liter)')
plt.ylabel('life expectancy (year)')
plt.title ('Scatterplot for the Association Between alcohol consumption and life expectancy')
print(scat1)

print ("OLS regression model for the association between alcohol consumption and life expectancy")
reg1 = smf.ols('lifeexpectancy ~ alcconsumption', data=sub2).fit()
print (reg1.summary())

