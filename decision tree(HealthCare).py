#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd


# In[19]:


import numpy as np


# In[20]:


from sklearn.model_selection import train_test_split


# In[64]:


from sklearn import metrics


# In[21]:


cr = pd.read_csv(r'E:\Dataset\train_2v.csv')


# In[22]:


cr.head(5)


# In[23]:


from sklearn.tree import DecisionTreeClassifier


# In[30]:


cr['smoking_status'].value_counts()


# In[29]:


cr.info()


# In[39]:


X=cr[['age','hypertension','heart_disease','avg_glucose_level','bmi']]


# In[40]:


y=cr[['stroke']]


# In[43]:


x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state=0)


# In[49]:


x_train = x_train.fillna(x_train.mean())


# In[50]:


y_train = y_train.fillna(y_train.mean())


# In[55]:


x_test = x_test.fillna(x_test.mean())


# In[56]:


y_test = y_test.fillna(y_test.mean())


# In[51]:


lm=DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)


# In[52]:


lm.fit(x_train, y_train)


# In[58]:


prediction=lm.predict(x_test)


# In[60]:


pred=pd.DataFrame(prediction)


# In[62]:


pred


# In[65]:


print(metrics.accuracy_score(prediction, y_test))


# In[ ]:





# In[ ]:




