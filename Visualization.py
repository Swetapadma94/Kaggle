#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from io import StringIO


# In[3]:


import io


# In[4]:


data=('a,b,c\n',
        '4,apple,bat\n',
         '8,orange,cow')


# In[5]:


data


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#When we use matplotlib basically we use plt.show(),instead of using all the time plt.show() directly we can run the code.


# In[7]:


import numpy as np


# In[8]:


x=np.arange(0,10)
y=np.arange(11,21)


# In[9]:


x


# In[10]:


y


# In[11]:


a=np.arange(40,50)
b=np.arange(50,60)


# In[12]:


a


# In[13]:


b


# In[14]:


# plotting using matplotlib
## plt Scatter
plt.scatter(x,y,c='g')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title(" 2D Graph")
plt.savefig('first.png')


# In[15]:


plt.plot(x,y,c='m')


# In[16]:


plt.plot(x,y,'g--')


# In[17]:


plt.plot(x,y,'b*-')


# In[18]:


plt.plot(x,y,'^k')


# In[19]:


plt.plot(x,y,'d')


# In[20]:


plt.plot(x,y,'y*',linestyle='dashed',linewidth=2,markersize=15)


# In[21]:


# creating subplot(within one canvas we can plot various plot)
plt.subplot(2,2,1)
plt.plot(x,y,'r')

plt.subplot(2,2,2)
plt.scatter(x,y,c='g')
plt.subplot(2,2,3)
plt.plot(x,y,'y*',linestyle='dashed',linewidth=2,markersize=15)
plt.subplot(2,2,4)
plt.plot(x,y,'bo')


# In[22]:


x=np.arange(1,10)
y=x*x
plt.plot(x,y,'g')


# In[23]:


np.pi


# In[24]:


#sign Diagram
x=np.arange(0,4*np.pi,0.1)
y=np.sin(x)
plt.plot(x,y,'*')


# In[25]:


x=np.arange(0,5*np.pi,0.1)
y_sin=np.sin(x)
y_cos=np.cos(x)

plt.subplot(2,1,1)
plt.plot(x,y_sin,'g*')
plt.title("sin plot",)
plt.subplot(2,1,2)
plt.plot(x,y_cos,'b*')
plt.title("cos plot")


# In[26]:


x=np.array([22,87,5,43,56,73,55,54,11,20,51,79,31,27])
plt.hist(x,bins=30)


# In[27]:


data=[np.random.normal(0,std,100) for std in range(1,4)]
plt.boxplot(data,vert=True,patch_artist=True)


# In[28]:


data


# In[29]:


labels='python','R','Tableau','Java'
sizes=[215,130,245,210]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0.1,0,0,0) #explode 1st slice
#plot
plt.pie(sizes,explode=explode,labels=labels,colors=colors,
autopct='%1.1f%%',shadow=True)
plt.axis('equal')

# Seaborn
.Dist plot-Univariate Analysis
.Joint plot-Bivariate Analysis
.Pair plot-for multivariate Analysis


# In[30]:


import seaborn as sns


# In[31]:


df=sns.load_dataset("tips")


# In[32]:


df.head()


# In[33]:


df.shape


# ## Heat Map-2D correlation matrix
# it shows correlation between ach n every features
# 

# In[38]:


df.info()


# In[37]:


df.corr()


# In[39]:


sns.heatmap(df.corr())


# In[41]:


#Join plot
sns.jointplot(x="tip",y="total_bill",data=df,kind='hex')


# In[42]:


#Join plot-pdf(probability density function)
sns.jointplot(x="tip",y="total_bill",data=df,kind='reg')


# In[44]:


#Join plot
sns.jointplot(x="tip",y="total_bill",data=df,kind='kde')


# In[45]:


#Join plot
sns.jointplot(x="tip",y="total_bill",data=df,kind='scatter')


# In[46]:


#pairplot-feature should be int or float
sns.pairplot(df)


# In[47]:


sns.pairplot(df,hue='sex')


# In[49]:


sns.pairplot(df,hue='smoker')


# In[50]:


df['smoker'].value_counts


# In[53]:


sns.distplot(df['tip'])


# In[54]:


sns.distplot(df['tip'],kde=False,bins=10)
#kde(kernel-density-Estimation)-False-count in y-axis


# #plot for categorical data
# 1.Box plot
# 2.violin plot
# 3.count plot
# 4.Bar plot

# In[55]:


## count plot-counting no. of observation in dataset
sns.countplot('sex',data=df)


# In[58]:


sns.countplot('time',data=df)


# In[59]:


sns.countplot('smoker',data=df)


# In[60]:


sns.countplot('day',data=df)


# In[62]:


sns.countplot(y='day',data=df)


# In[63]:


#Bar plot
sns.barplot(x='tip',y='time',data=df)


# In[64]:


#Bar plot
sns.barplot(x='total_bill',y='sex',data=df)


# In[65]:


#Bar plot
sns.barplot(y='tip',x='time',data=df)


# In[66]:


sns.boxplot('smoker','total_bill',data=df)


# In[68]:


sns.boxplot(x='sex',y='total_bill',data=df,palette='rainbow')


# In[70]:


sns.boxplot(data=df,orient='v')


# In[73]:


sns.boxplot(x='total_bill',y='day',hue='smoker',data=df)


# In[75]:


# Violin plot(combination of boxplot and kde)
sns.violinplot(x='total_bill',y='day',hue='sex',data=df,palette='rainbow')


# In[76]:


# Violin plot(combination of boxplot and kde)
sns.violinplot(x='total_bill',y='day',data=df,palette='rainbow')


# In[77]:


data=sns.load_dataset('iris')


# In[78]:


data.head()


# In[ ]:




