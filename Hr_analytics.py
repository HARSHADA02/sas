#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


# In[2]:


train=pd.read_csv(r"C:\Users\user30\Downloads\Hr_train.csv")
test=pd.read_csv(r"C:\Users\user30\Downloads\Hr_test.csv")


# In[3]:


train.head()


# In[4]:


train['is_promoted'].value_counts()


# In[5]:


train.info()


# In[6]:


train.describe()


# In[7]:


print(train.isnull().sum())


# In[8]:


train.shape


# In[9]:


test.shape


# In[10]:


train.dropna(inplace=True)


# In[11]:


test.dropna(inplace=True)


# In[12]:


train.shape


# In[13]:


test.shape


# In[14]:


x=train[['no_of_trainings','length_of_service','avg_training_score']]
y=train[['is_promoted']]


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3, random_state=0)


# In[ ]:





# In[ ]:





# In[17]:


model=LogisticRegression()


# In[18]:


model.fit(x_train,y_train)


# In[19]:


prediction=model.predict(x_test)


# In[20]:


pred=pd.DataFrame(prediction)


# In[21]:


pred


# In[22]:


from sklearn import metrics


# In[23]:


data1=metrics.confusion_matrix(y_test,prediction)


# In[24]:


data1


# In[25]:


print(metrics.accuracy_score(y_test,prediction))

