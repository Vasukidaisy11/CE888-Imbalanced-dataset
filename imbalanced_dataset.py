#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import NaN
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import datasets


# In[52]:


#Load the first dataframe
dataset_1 = pd.read_csv("WineQT.csv")
df_1 = dataset_1.copy()


# In[53]:


#Display the data info
df_1.info()


# In[54]:


#checking for missing values
df_1=df_1.drop(['Id'],axis=1)
df_1.isnull().sum()


# In[55]:


#Visualization of dataset_1
sns.countplot(data=df_1)


# In[56]:


#SPlit the data set in target variable and features
features=df_1.drop(['quality'],axis=1)
target_var=df_1['quality']
target_var


# In[57]:


# low imbalance of 65% in dataset_1 
low_imbalance_dataset_1 = features
low_imbalance_dataset_1.density.iloc[0:743] = NaN
low_imbalance_dataset_1.info()


# In[58]:


# Visualization of low_imbalance_dataset_1 
sns.countplot(data=low_imbalance_dataset_1)


# In[59]:


# # Medium imbalance of 75% in dataset_1 
medium_imbalance_dataset_1 = features
medium_imbalance_dataset_1.density.iloc[0:857] = NaN
medium_imbalance_dataset_1.info()


# In[60]:


# Visualization of medium_imbalance_dataset_1
sns.countplot(data=medium_imbalance_dataset_1)


# In[61]:


# High imbalance of 90% in dataset_1 
high_imbalance_dataset_1 = features
high_imbalance_dataset_1.density.iloc[0:1028] = NaN
high_imbalance_dataset_1.info()


# In[62]:


# Visualization of High_imbalance_dataset_1
sns.countplot(data=high_imbalance_dataset_1)


# In[67]:


# Stratified cross-validation
def clean_df(df):
    assert isinstance(df, pd.Df), 
    df.dropna(inplace=True)
    index = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[index].astype(np.float64)


# In[70]:


skf = StratifiedKFold(n_splits=10)


# In[88]:



x = features
y = target_var


# In[89]:


def training(train, test, fold_no):
  x_train = train.drop(['density'],axis=1)
  y_train = train.Outcome
  x_test = test.drop(['density'],axis=1)
  y_test = test.Outcome
  model.fit(x_train, y_train)
  score = model.score(x_test,y_test)


# In[86]:


num = 1
for train_index,test_index in skf.split(x, y):
  train = df_1.iloc[train_index,:]
  test = df_1.iloc[test_index,:]
  training(train, test, no)
  nnum += 1


# In[87]:


# Random Forest


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:



regressor = RandomForestRegressor(n_esti=20, random_state=0)
regressor.fit(X_train, y_train)
y_pre = regressor.predict(X_test)
print('Mean Absolute Error:', df_1.mean_absolute_error(y_test, y_pre))


# In[ ]:


#/////////////////////////////////////////////////////


# In[ ]:


#Load the second dataframe
dataset_2 = pd.read_csv("avocado.csv")
df_2 = dataset_2.copy()

#Display the data info
df_2.info()

#checking for missing values
df_2.isnull().sum()

#Visualization of dataset_1
sns.countplot(data=df_2)

#SPlit the data set in target variable and features
features=df_2.drop(['quality'],axis=1)
target_var=df_2['quality']
target_var

# low imbalance of 65% in dataset_2 
low_imbalance_dataset_2 = features
low_imbalance_dataset_2.density.iloc[0:743] = NaN
low_imbalance_dataset_2.info()

# Visualization of low_imbalance_dataset_1 
sns.countplot(data=low_imbalance_dataset_1)

# # Medium imbalance of 75% in dataset_2 
medium_imbalance_dataset_2 = features
medium_imbalance_dataset_2.density.iloc[0:857] = NaN
medium_imbalance_dataset_2.info()

# Visualization of medium_imbalance_dataset_2
sns.countplot(data=medium_imbalance_dataset_1)

# High imbalance of 90% in dataset_2
high_imbalance_dataset_2 = features
high_imbalance_dataset_2.density.iloc[0:1028] = NaN
high_imbalance_dataset_2.info()

# Visualization of High_imbalance_dataset_2
sns.countplot(data=high_imbalance_dataset_2)

# Stratified cross-validation
def clean_df(df):
    assert isinstance(df, pd.Df), 
    df.dropna(inplace=True)
    index = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[index].astype(np.float64)

skf = StratifiedKFold(n_splits=10)


x = features
y = target_var


def training(train, test, fold_no):
  x_train = train.drop(['density'],axis=1)
  y_train = train.Outcome
  x_test = test.drop(['density'],axis=1)
  y_test = test.Outcome
  model.fit(x_train, y_train)
  score = model.score(x_test,y_test)

num = 1
for train_index,test_index in skf.split(x, y):
  train = df_2.iloc[train_index,:]
  test = df_2.iloc[test_index,:]
  training(train, test, no)
  nnum += 1

# Random Forest

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = RandomForestRegressor(n_esti=20, random_state=0)
regressor.fit(X_train, y_train)
y_pre = regressor.predict(X_test)
print('Mean Absolute Error:', df_2.mean_absolute_error(y_test, y_pre))


# In[ ]:


#Load the third dataframe
dataset_3 = pd.read_csv("winequality-red.csv")
df_3 = dataset_3.copy()

#Display the data info
df_3.info()

#checking for missing values
df_3.isnull().sum()

#Visualization of dataset_3
sns.countplot(data=df_3)

#SPlit the data set in target variable and features
features=df_3.drop(['quality'],axis=1)
target_var=df_3['quality']
target_var

# low imbalance of 65% in dataset_3 
low_imbalance_dataset_3 = features
low_imbalance_dataset_3.density.iloc[0:743] = NaN
low_imbalance_dataset_3.info()

# Visualization of low_imbalance_dataset_3 
sns.countplot(data=low_imbalance_dataset_3)

# # Medium imbalance of 75% in dataset_3 
medium_imbalance_dataset_3 = features
medium_imbalance_dataset_3.density.iloc[0:857] = NaN
medium_imbalance_dataset_3.info()

# Visualization of medium_imbalance_dataset_3
sns.countplot(data=medium_imbalance_dataset_3)

# High imbalance of 90% in dataset_3
high_imbalance_dataset_3 = features
high_imbalance_dataset_3.density.iloc[0:1028] = NaN
high_imbalance_dataset_3.info()

# Visualization of High_imbalance_dataset_3
sns.countplot(data=high_imbalance_dataset_3)

# Stratified cross-validation
def clean_df(df):
    assert isinstance(df, pd.Df), 
    df.dropna(inplace=True)
    index = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[index].astype(np.float64)

skf = StratifiedKFold(n_splits=10)


x = features
y = target_var


def training(train, test, fold_no):
  x_train = train.drop(['density'],axis=1)
  y_train = train.Outcome
  x_test = test.drop(['density'],axis=1)
  y_test = test.Outcome
  model.fit(x_train, y_train)
  score = model.score(x_test,y_test)

num = 1
for train_index,test_index in skf.split(x, y):
  train = df_3.iloc[train_index,:]
  test = df_3.iloc[test_index,:]
  training(train, test, no)
  nnum += 1

# Random Forest

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = RandomForestRegressor(n_esti=20, random_state=0)
regressor.fit(X_train, y_train)
y_pre = regressor.predict(X_test)
print('Mean- Absolute-Error:', df_3.mean_absolute_error(y_test, y_pre))


# In[ ]:


# K- Means algorithm

import numpy as np
from numpy.linalg import norm


class K_means:

    def __init__(self, n, max_iter=100, random_state=123):
        self.n = n
        self.max = max
        self.random = random

    def centroids(self, X):
        np.random.RandomState(self.random)
        idx = np.random.permutation(X.shape[0])
        cent = X[random_idx[:self.n_clusters]]
        return cent

    def cent(self, X, l):
        cent = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[l == k, :], axis=0)
        return cent
    
    def distance(self, X, centroids):
        d = np.zeros((X.shape[0], self.n_clusters))
        for Z in range(self.n_clusters):
            row_n = norm(Z - centroids[k, :], axis=1)
            d[:, k] = np.square(row_n)
        return d

    def find_closest_cluster(self, d):
        return np.argmin(d, axis=1)

    def c_sse(self, X, l, cent):
        d = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            d[l == k] = norm(X[l == k] - cent[k], axis=1)
        return np.sum(np.square(d))
    
    def fit(self, X):
        self.c = self.initialize_c(X)
        for i in range(self.max_iter):
            old_cent = self.cent
            d= self.distance(X, old_cent)
            self.l = self.find_closest_cluster(d)
            self.cent = self.centroids(X, self.l)
            if np.all(old_cent == self.cent):
                break
        self.error = self.compute_sse(X, self.l, self.cent)
    
    def predict(self, X):
        d = self.distance(X, self.cent)
        return self.find_closest_cluster(d)
kmeans = K_means(num_clusters=2, maximum_iterration=100)
kmeans.fit(X_std)
cent = kmeans.cent


#Elbow method


sse = []
list_k = list(range(1, 10))

for x in list_k:
    km = KMeans(n_clusters=x)
    km.fit(X_std)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




