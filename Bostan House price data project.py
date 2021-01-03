#!/usr/bin/env python
# coding: utf-8

# ## Real Estate Problem

# In[1]:


#import pandas library
import pandas as pd


# In[2]:


#Boston city dataset
data=pd.read_csv("data.csv")
data.head()# it gives five row of your dataset.


# In[3]:


#CRIM -per capita crime rate by town.
#ZN -proportion of residential land zoned for lots over 25,000 sq.ft.
#INDUS -proportion of non-retail business acres per town.
#CHAS- Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
#NOX- nitrogen oxides concentration (parts per 10 million).
#RM- average number of rooms per dwelling.
#AGE- proportion of owner-occupied units built prior to 1940.
#DIS- weighted mean of distances to five Boston employment centres.
#RAD- index of accessibility to radial highways.
#TAX- full-value property-tax rate per \$10,000.
#PTRATIO- pupil-teacher ratio by town.
#B- 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
#LSTAT- lower status of the population (percent).

##TARGET VARIABLE(MEDV)
#MEDV- median value of owner-occupied homes in \$1000s.


# In[4]:


#it gives all information of your dataset.
data.info()


# In[5]:


#it gives all statistics information.
data.describe() # σ = sqrt [ Σ ( Xi - μ )2 / N ]


# In[6]:


data['CHAS'].value_counts()


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


#ploting the histogram of all dataset individualy.
data.hist(bins=50,figsize=(20,15))
#using histogram all data Analyse very eaisly.


# ## Split the Data

# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)


# In[11]:


len(train_set)


# In[12]:


train_set.shape


# In[13]:


len(test_set)


# In[14]:


test_set.shape


# In[15]:


data["CHAS"].value_counts()


# In[16]:


#train and test data both present CHAS
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(data,data['CHAS']):
    strat_train_set=data.loc[train_index]
    strat_test_set=data.loc[test_index]


# In[17]:


strat_train_set['CHAS'].value_counts()


# In[18]:


strat_train_set.shape


# In[19]:


strat_test_set.shape


# In[20]:


strat_test_set['CHAS'].value_counts()


# In[21]:


print(f"equal distribute of train {376/28} and test {95/7}")


# In[22]:


data = strat_train_set.copy()


# ## Knowing the Correlarion

# In[23]:


#When two sets of data are strongly linked together we say they have a High Correlation.
#1 is a perfect positive correlation.
#0 is no correlation.
#-1 is a perfect negative correlation.
corr_matrix=data.corr()
corr_matrix


# In[24]:


#MEDV is a Dependent or Target variable
corr_matrix['MEDV'].sort_values(ascending=False)


# In[25]:


from pandas.plotting import scatter_matrix
attributs=['MEDV','RM','ZN','LSTAT']
scatter_matrix(data[attributs],figsize=(12,8))


# In[26]:


plt.scatter(x=data['RM'],y=data['MEDV'])
plt.xlabel('average no of room')
plt.ylabel('price')
plt.show()


# ## Trying out Attribute combination

# In[27]:


data['TAXRM']=data['TAX']/data['RM']
data['TAXRM']


# In[28]:


data.head()


# In[29]:


corr_matrix=data.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[30]:


plt.scatter(x=data['TAXRM'],y=data['MEDV'])
plt.xlabel('TAXRM')
plt.ylabel('price')
plt.show()
#heighly correlarion showing.


# In[31]:


data = strat_train_set.drop("MEDV", axis=1)#independent variable or targrt variable
data_y = strat_train_set["MEDV"].copy() # dependent variable or label variable
print(data.shape)
print(data_y.shape)


# ## Missing value Attribute

# In[32]:


#handling of missing value in diffrent ways
#1-Deleting Rows or Deleting columns
#2-Replacing With Mean/Median/Mode
#3-Assigning An Unique Category
#4-Using Algorithms Which Support Missing Values


# In[33]:


#inplace of missing value replace with it's median value
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit(data)


# In[34]:


#to calculate all columns median value 
imputer.statistics_


# In[35]:


#transformed data frame without missing values
X=imputer.transform(data)
data_tr=pd.DataFrame(X,columns=data.columns)
data_tr.describe()


# ## Feature Scaling
# 1-Min Max Scaler(Normalization)
# (value - min)/(max - min) range 0 to 1
# 2-Standardization
# (value - mean)/std

# ## creating a pipelines

# In[36]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),
                      #add how want to create pipeline
                     ('std_scaler',StandardScaler()),
                     ])


# In[37]:


data_tr_num=my_pipeline.fit_transform(data_tr)
data_tr_num


# In[38]:


data_tr_num.shape


# ## selecting desired model for Real Estate Problem
# 

# ## Linear Regression model

# In[39]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(data_tr_num,data_y)


# In[40]:


some_data=data.iloc[:5]
some_label=data_y.iloc[:5]


# In[41]:


prepared_data=my_pipeline.transform(some_data)


# In[42]:


#prediction value
linreg.predict(prepared_data)


# In[43]:


#Actual value
list(some_label)


# In[44]:


import numpy as np


# ## Evalueating the Model

# In[45]:


from sklearn.metrics import mean_squared_error
prediction_value=linreg.predict(data_tr_num)
mse=mean_squared_error(data_y,prediction_value)
rmse=np.sqrt(mse)
mse,rmse


# In[46]:


#Score of your model 
linreg.score(data_tr_num,data_y)


# ## Using Desion Tree Regression model

# In[47]:


from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(data_tr_num,data_y)


# In[48]:


some_data=data.iloc[:5]
label_data=data_y.iloc[:5]


# In[49]:


prepared_data2=my_pipeline.transform(some_data)


# In[50]:


model.predict(prepared_data2)


# In[51]:


list(label_data)


# In[52]:


#overfit the traning data
from sklearn.metrics import mean_squared_error
prediction_value=model.predict(data_tr_num)
mse=mean_squared_error(data_y,prediction_value)
rmse=np.sqrt(mse)
mse,rmse


# ## Random Forest Model

# In[53]:


from sklearn.ensemble import RandomForestRegressor
rand_model=RandomForestRegressor()
rand_model.fit(data_tr_num,data_y)


# In[54]:


some_data=data.iloc[:5]
label_data=data_y.iloc[:5]


# In[55]:


prepared_data3=my_pipeline.transform(some_data)


# In[56]:


rand_model.predict(prepared_data3)


# In[57]:


list(label_data)


# In[58]:


from sklearn.metrics import mean_squared_error
prediction_value=rand_model.predict(data_tr_num)
mse=mean_squared_error(data_y,prediction_value)
rmse=np.sqrt(mse)
mse,rmse


# ## Using better Evaluation Technique-Cross Validation

# In[59]:


#for Desion Tree
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data_tr_num, data_y, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores


# In[60]:


rmse_scores.mean(),rmse_scores.std()


# In[61]:


#for Linear Regreesion
from sklearn.model_selection import cross_val_score
scores = cross_val_score(linreg, data_tr_num, data_y, scoring="neg_mean_squared_error", cv=10)
rmse_scores_linreg = np.sqrt(-scores)
rmse_scores_linreg


# In[62]:


rmse_scores_linreg.mean(),rmse_scores_linreg.std()


# In[63]:


#for random forest model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rand_model, data_tr_num, data_y, scoring="neg_mean_squared_error", cv=10)
rmse_scores_rand = np.sqrt(-scores)
rmse_scores_rand


# In[64]:


rmse_scores_rand.mean(),rmse_scores_rand.std()
#we choose the random forest model


# ## Saving the Models

# In[65]:


from joblib import load,dump
dump(rand_model,'Dragon.joblib')


# ## Testing The Models

# In[66]:


X_test=strat_test_set.drop('MEDV',axis=1)
y_test=strat_test_set['MEDV']
X_test_prepared=my_pipeline.transform(X_test)


# In[67]:


final_prediction=rand_model.predict(X_test_prepared)
mse=mean_squared_error(final_prediction,y_test)
final_rmse=np.sqrt(mse)
final_rmse


# In[68]:


final_prediction


# In[69]:


np.array(y_test)


# ## Using this Model

# In[70]:


from joblib import dump, load
import numpy as np
model = load('Dragon.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
model.predict(features)


# In[ ]:




