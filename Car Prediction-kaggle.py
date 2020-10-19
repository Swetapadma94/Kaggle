#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\car\car data.csv',encoding='latin1')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print(df['Seller_Type'].unique())


# In[6]:


print(df['Transmission'].unique())


# In[7]:


print(df['Owner'].unique())


# In[8]:


df.isna().sum()


# In[9]:


df.describe


# In[10]:


df.info()


# In[11]:


df.columns


# In[13]:


final_dataset=df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[14]:


final_dataset


# In[15]:


final_dataset['Current_year']=2020


# In[16]:


final_dataset.head()


# In[17]:


final_dataset['no_year']=final_dataset['Current_year']-final_dataset['Year']


# In[18]:


final_dataset.head()


# In[19]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[21]:


final_dataset.drop(['Current_year'],axis=1,inplace=True)


# In[22]:


final_dataset


# In[23]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[25]:


final_dataset.head()


# In[26]:


import seaborn as sns


# In[27]:


sns.pairplot(final_dataset)


# In[28]:


sns.heatmap(final_dataset)


# In[32]:



import seaborn as sns
#get correlations of each features in dataset
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[33]:


X=final_dataset.iloc[:,1:]


# In[39]:


X.head()


# In[37]:


y=final_dataset.iloc[:,0]


# In[40]:


y.head()


# In[45]:



### Feature Importance

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[47]:


print(model.feature_importances_)


# In[48]:


#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[51]:


X_train.shape


# In[54]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# In[55]:



n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[57]:



from sklearn.model_selection import RandomizedSearchCV


# In[58]:



#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[59]:



# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[61]:


rf=RandomForestRegressor()


# In[62]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[63]:


rf_random.fit(X_train,y_train)


# In[64]:


rf_random.best_params_


# In[65]:


rf_random.best_score_


# In[67]:


prediction=rf_random.predict(X_test)


# In[68]:


prediction


# In[71]:


sns.distplot(y_test-prediction)


# In[73]:


plt.scatter(y_test,prediction)


# In[74]:



from sklearn import metrics


# In[76]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[77]:



import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)


# In[80]:


with open('random_forest_regression_model.pkl', 'rb') as handle:
    b = pickle.load(handle)

print (file == b)


# In[ ]:




