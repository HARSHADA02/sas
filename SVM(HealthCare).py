#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn import metrics


# In[9]:


cr = pd.read_csv(r'E:\Dataset\train_2v.csv')


# In[10]:


cr.head(5)


# In[13]:


from sklearn import svm


# In[14]:


cr['smoking_status'].value_counts()


# In[15]:


cr.info()


# In[16]:


X=cr[['age','hypertension','heart_disease','avg_glucose_level','bmi']]


# In[17]:


y=cr[['stroke']]


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state=0)


# In[19]:


x_train = x_train.fillna(x_train.mean())


# In[20]:


y_train = y_train.fillna(y_train.mean())


# In[21]:


x_test = x_test.fillna(x_test.mean())


# In[22]:


y_test = y_test.fillna(y_test.mean())


# In[23]:


lm=svm.SVC(kernel='linear')


# In[25]:


lm.fit(x_train, y_train)


# In[27]:


prediction=lm.predict(x_test)


# In[29]:


pred=pd.DataFrame(prediction)
pred


# In[30]:


print(metrics.accuracy_score(prediction, y_test))

