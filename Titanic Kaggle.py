#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\train.csv',encoding='latin1')
test=pd.read_csv(r'E:\Krish naik\kaggle dataset\Titanic\test.csv',encoding='latin1')


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


print(train.shape)
print(test.shape)


# In[6]:


train.describe


# In[7]:


test.columns


# In[8]:


train.isna().sum()


# In[9]:


train.describe(include=['O'])


# In[10]:


train.info()


# In[11]:


numerical = [col for col in train.columns if train[col].dtypes != 'O']

numerical


# In[12]:


numerical = [col for col in train.columns if train[col].dtypes == 'O']

numerical


# In[13]:


train.Survived.value_counts()


# In[14]:


train.Pclass.value_counts()


# In[15]:


train.groupby('Pclass').Survived.value_counts()


# In[16]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[17]:


train.groupby('Sex').Survived.value_counts()


# In[18]:


train.groupby('Age').Survived.value_counts()


# In[19]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()


# In[20]:


sns.heatmap(train.isna(),yticklabels=False,cmap='viridis')


# In[21]:


sns.heatmap(train.isna(),yticklabels=False,cmap='mako')


# In[22]:


sns.heatmap(train.isna())


# In[23]:


sns.heatmap(train.isna(),yticklabels=False,cmap='summer')


# In[24]:


sns.countplot(x='Survived',data=train)


# In[25]:


sns.set_style("whitegrid")
sns.countplot(x='Survived',hue='Sex',data=train)


# In[26]:


sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[27]:


sns.set_style("whitegrid")
sns.countplot(x='Survived',hue='Age',data=train)


# In[28]:


sns.distplot(train['Age'],kde=False,color='green',bins=20)


# In[29]:


sns.distplot(train['Survived'],kde=False,color='green',bins=20)


# In[30]:


sns.countplot(x="SibSp",data=train)


# In[31]:


sns.distplot(train["Fare"],kde=False,color='green')


# In[32]:


sns.distplot(train["Pclass"],kde=False,color='green')


# In[33]:


sns.countplot(x='Pclass',data=train)


# In[34]:


plt.figure(figsize=(12,10))
sns.boxplot(x='Pclass',y='Age',data=train)


# In[35]:


sns.boxplot(data=train)


# In[36]:


sns.pairplot(train)


# In[37]:


sns.barplot(x='Survived',y='Pclass',data=train)


# In[38]:


sns.barplot(x='Sex',y='Survived',data=train)


# In[39]:


plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)


# In[40]:


train.isna().mean().sort_values(ascending=False)


# In[41]:


train.isna().sum().sort_values(ascending=False)


# In[42]:


train.head()


# # Imputation 

# In[43]:


train.fillna(train['Age'].median(),inplace=True)


# In[44]:


train.Age.isna().sum()


# In[45]:


# Cabin is not so important for our model and it has many NaN values,so we can drop that column
train.drop('Cabin',axis=1,inplace=True)


# In[46]:


train.columns


# In[47]:


train.fillna(train['Embarked'].mode(),inplace=True)


# In[48]:


train.isna().sum()


# In[49]:


features_drop = ['Name', 'Ticket']


# In[50]:


train.drop(features_drop,axis=1)


# In[51]:


train.head()


# In[52]:


features_drop = ['SibSp', 'Parch']


# In[53]:


train.drop(features_drop,axis=1,inplace=True)


# In[54]:


train.head()


# In[55]:


train.drop('PassengerId',axis=1,inplace=True)


# In[56]:


train.head()


# In[57]:


train['Sex'] = pd.get_dummies(train['Sex'],drop_first=True)

train['Embarked'] = pd.get_dummies(train['Embarked'],drop_first=True)


# In[58]:


train


# In[59]:


test.head()


# In[96]:


test.isna().sum()


# In[100]:


test.fillna(test['Age'].median(),inplace=True)


# In[101]:


test.fillna(test['Embarked'].mode(),inplace=True)


# In[102]:


drop=['Name','SibSp','Parch','Cabin']


# In[ ]:


test.drop(drop,axis=1,inplace=True)


# In[104]:


test


# In[63]:


test['Sex']=pd.get_dummies(test['Sex'],drop_first=True)
test['Embarked']=pd.get_dummies(test['Embarked'],drop_first=True)


# In[64]:


test.head()


# In[80]:


train.drop('Name',axis=1,inplace=True)


# In[87]:


train.drop('Ticket',axis=1,inplace=True)


# In[88]:


test.drop('Ticket',axis=1,inplace=True)


# In[89]:


print(train.shape)
test.shape


# In[91]:


train.columns


# In[92]:


test.columns


# In[105]:


X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()

X_train.shape, y_train.shape, X_test.shape


# In[106]:


# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


# # Logistic Regression

# In[107]:


clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' percent')


# In[108]:


clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_svc)


# In[109]:


clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print (acc_linear_svc)


# In[110]:


clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn)


# In[111]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree = clf.predict(X_test)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print (acc_decision_tree)


# In[112]:


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred_random_forest = clf.predict(X_test)
acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)
print (acc_random_forest)


# In[113]:


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print (acc_gnb)


# In[114]:


clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
print (acc_sgd)


# In[116]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent'],
    
    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_sgd]
    })

models.sort_values(by='Score', ascending=False)


# In[117]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred_random_forest
    })

 submission.to_csv('submission.csv', index=False)


# In[118]:


submission


# In[119]:


submission.to_csv('E:\Krish naik\kaggle dataset\Titanic\submission.csv', index=False)


# In[ ]:




