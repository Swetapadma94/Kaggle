#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[11]:


house=pd.read_csv(r'E:\Krish naik\kaggle dataset\House\train.csv',encoding='latin1',usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])


# In[16]:


house.columns


# In[12]:


house.head()


# In[18]:


house.shape


# ### How to handle Categorical Missing values
# 1.Frequnt Category Imputation.
# we shouldn't use this technique for that variable which is having more number of missing values.

# In[13]:


house.info


# In[14]:


house.isnull().sum()


# In[17]:


house.isnull().mean().sort_values(ascending=True)


# Here BsmtQual and GarageType has less number of missing values so we will replace NaN values with most frequnt value
# Compute the Frequency  with every feature.

# In[23]:


house.groupby(['BsmtQual'])['BsmtQual'].count().sort_values(ascending=False).plot.bar()


# In[24]:


house['BsmtQual'].value_counts().sort_values(ascending=False).plot.bar()


# In[25]:


house['GarageType' ].value_counts().sort_values(ascending=False).plot.bar()


# In[28]:


house.groupby(['FireplaceQu'])['FireplaceQu'].count().sort_values(ascending=False).plot.bar()


# In[30]:


house['BsmtQual'].value_counts().index[0]


# In[ ]:





# In[37]:


# Replacing function
def impute_nan(df,variable):
    most_frequnt_category=df[variable].value_counts().index[0]
    df[variable].fillna(most_frequnt_category,inplace=True)
    


# In[38]:


for feature in ['BsmtQual','FireplaceQu','GarageType']:
    impute_nan(house,feature)


# In[40]:


house.isnull().sum()


# #Advantages
# 1.Easy to implement.
# 
# 2.Faster way to implement.
# ##Disadvantages
# 1.Since we r using more frequent level,it may use in an overpresented if there are many NaN.
# 
# 2.It distorts the relation of the most frequent level.

# Adding a variable to capture NAN

# In[41]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\House\train.csv',encoding='latin1',usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])


# In[42]:


df.head()


# In[43]:


df.isna().sum()


# In[50]:


df['BsmtQual_var']=np.where(df['BsmtQual'].isnull(),1,0)


# In[52]:


df.head()


# In[49]:


df.isna().sum()


# In[56]:


freq=df['BsmtQual'].mode()[0]
freq


# In[58]:


df['BsmtQual'].fillna(freq,inplace=True)


# In[59]:


df['BsmtQual'].isnull().sum()


# In[67]:


df['FireplaceQu_var']=np.where(df['FireplaceQu'].isnull(),1,0)


# In[68]:


df.head()


# #### if we have mor frequent category,we just replace NaN with a new category
# 
# 

# In[72]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\House\train.csv',encoding='latin1',usecols=['BsmtQual','FireplaceQu','GarageType','SalePrice'])


# In[73]:


df.head()


# In[90]:


def impute_nan(df,variable):
    df[variable+'_newvar']=np.where(df[variable].isnull(),'missing',df[variable])


# In[91]:


for feature in ['BsmtQual','FireplaceQu','GarageType']:
    impute_nan(df,feature)


# In[92]:


df.head()


# In[78]:


df.isnull().sum()


# In[95]:


df=df.drop(['BsmtQual','FireplaceQu','GarageType'],axis=1)


# In[96]:


df


# Handling Categorical Features
# One Hot Encoding

# In[1]:


import pandas as pd


# In[4]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Sex'])


# In[3]:


df.head()


# In[5]:


df.head()


# In[7]:


pd.get_dummies(df,drop_first=True).head()


# In[8]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1',usecols=['Embarked'])


# In[9]:


df['Embarked'].unique()


# In[11]:


df.dropna(inplace=True)


# In[12]:


df.head()


# In[16]:


pd.get_dummies(df,drop_first=True)


# -->Disadvantages:
# 1.it will create many number of features.
# 2.Curs of dimensoinality.

# -->How to deal with many categories-(One Hot Encoding).
# --->Benz Data set

# In[19]:


df=pd.read_csv(r'E:\Krish naik\kaggle dataset\Mercedes\train.csv',encoding='latin1',usecols=['X0','X1','X2','X3','X4','X5','X6'])


# In[20]:


df.head()


# In[21]:


df.isna().sum()


# In[22]:


for i in df.columns:
    print(df[i].value_counts)


# In[24]:


df['X0'].value_counts()


# In[30]:


for i in df.columns:
    print(len(df[i].unique()))


# #Ten most frequent category (KDD-Orange cup)

# In[32]:


df.X1.value_counts().sort_values(ascending=False).head(10)


# In[33]:


list_ten=df.X1.value_counts().sort_values(ascending=False).head(10).index


# In[35]:


list_ten=list(list_ten)


# In[36]:


list_ten


# In[37]:


for categories in list_ten:
    df[categories]=np.where(df['X1']==categories,1,0)


# In[38]:


df


# In[41]:


list_ten.append('X1')


# In[43]:


df[list_ten]


# In[44]:


df.head() 


# In[ ]:




