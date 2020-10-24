#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Display all the columns of the data frame
pd.pandas.set_option('display.max_columns',None)


# In[3]:


data=pd.read_csv(r'E:\Krish naik\kaggle dataset\House\train.csv',encoding='latin1')


# In[4]:


data.head()


# In[5]:


data.shape


# # In Data Analysis We will Analyze To Find out the below stuff¶
# Missing Values
# All The Numerical Variables
# Distribution of the Numerical Variables
# Categorical Variables
# Cardinality of Categorical Variables
# Outliers
# Relationship between independent and dependent feature(SalePrice)

# In[6]:


##finding missing values
data.isna().sum()


# In[8]:


## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in data.columns if data[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(data[feature].isnull().mean(), 4),  ' % missing values')


# Since they are many missing values, we need to find the relationship between missing values and Sales Price¶
# Let's plot some diagram for this relationship

# In[9]:


for feature in features_with_na:
    df = data.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    df[feature] = np.where(df[feature].isnull(), 1, 0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    df.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# In[10]:


df.columns


# In[11]:


len(df.Id)


# In[14]:


# find the numerical column
numerical=[col for col in data.columns if data[col].dtypes !="O"]
print('Number of numerical variables: ', len(numerical))

# visualise the numerical variables
data[numerical].head()


# In[17]:


for i in numerical:
    print(data[i].value_counts())


# Temporal Variables(Eg: Datetime Variables)¶
# From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold.

# In[18]:


# list of variables that contain year information
year_feature = [feature for feature in numerical if 'Yr' in feature or 'Year' in feature]

year_feature


# In[19]:


# let's explore the content of these year variables
for feature in year_feature:
    print(feature, data[feature].unique())


# In[21]:


## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price
data.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# In[23]:


## Here we will compare the difference between All years feature with SalePrice

for feature in year_feature:
    
        df=data.copy()
    

        plt.scatter(df[feature],df['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[24]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical if len(data[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[25]:


discrete_feature


# In[26]:


data[discrete_feature].head()


# In[28]:


## Lets Find the realtionship between them and Sale PRice

for feature in discrete_feature:
    df=data.copy()
    df.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# ## There is a relationship between variable number and SalePrice

# In[29]:


continuous_feature=[feature for feature in numerical if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[31]:


for feature in continuous_feature:
    df=data.copy()
    df[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# # Exploratory Data Analysis Part 2

# In[34]:


## We will be using logarithmic transformation
for feature in continuous_feature:
    df=data.copy()
    if 0 in df[feature].unique():
        pass
    else: 
        df[feature]=np.log(df[feature])
        df['SalePrice']=np.log(df['SalePrice'])
        plt.scatter(df[feature],df['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()


# # Outliers

# In[38]:


for feature in continuous_feature:
    df=data.copy()
    if 0 in df[feature].unique():
        pass
    else: 
        df[feature]=np.log(df[feature])
        df.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# In[43]:


# Categorical Features
categorical=[col for col in data.columns if data[col].dtypes=="O"]
categorical


# In[46]:


data[categorical].head()


# In[47]:


for feature in categorical:
    print('The feature is {} and number of categories are {}'.format(feature,len(data[feature].unique())))


# In[49]:


for feature in categorical:
    df=data.copy()
    df.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:




