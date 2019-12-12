#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


from sklearn import metrics


# In[5]:


cr = pd.read_csv(r'E:\Dataset\train_2v.csv')


# In[6]:


cr.head(5)


# In[7]:


cr['smoking_status'].value_counts()


# In[8]:


cr.info()


# In[9]:


X=cr[['age','hypertension','heart_disease','avg_glucose_level','bmi']]


# In[10]:


y=cr[['stroke']]


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state=0)


# In[17]:


x_train = x_train.fillna(x_train.mean())


# In[18]:


y_train = y_train.fillna(y_train.mean())


# In[19]:


x_test = x_test.fillna(x_test.mean())


# In[20]:


y_test = y_test.fillna(y_test.mean())


# In[12]:


from sklearn import ensemble 


# In[14]:


params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}


# In[15]:


model=ensemble.GradientBoostingRegressor(**params)


# In[21]:


model.fit(x_train, y_train)


# In[22]:


prediction=model.predict(x_test)


# In[24]:


pred=pd.DataFrame(prediction)
pred


# In[27]:


model_score=model.score(x_train, y_train)
print(model_score)


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


fig, ax = plt.subplots()
ax.scatter(y_test, prediction, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()

