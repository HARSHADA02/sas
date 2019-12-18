#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score


# In[33]:


cr = pd.read_csv(r'E:\Python Datascience\noshowappointments\healthspring.csv')


# In[34]:


cr.head()


# In[35]:


cr['No-show'].value_counts()


# In[36]:


cr['Neighbourhood'].value_counts()


# In[37]:


cr.info()


# In[38]:


def score_to_numeric(x):
    if x=='Yes':
        return 1
    if x=='No':
        return 0
cr['No-show'] = cr['No-show'].apply(score_to_numeric)


# In[39]:


X=cr[['SMS_received','Age']]


# In[40]:


y=cr[['No-show']]


# In[41]:


x_train, x_val, y_train, y_val = train_test_split(X,y,
                                                  test_size = 0.1,
                                                  random_state=0)


# In[42]:


from sklearn import metrics


# In[43]:



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}


# In[44]:


from sklearn import ensemble 


# In[45]:



model=ensemble.GradientBoostingRegressor(**params)


# In[47]:


model.fit(x_train, y_train)


# In[50]:


prediction=model.predict(x_val)


# In[52]:



pred=pd.DataFrame(prediction)


# In[54]:


pred


# In[56]:



import matplotlib.pyplot as plt


# In[58]:



from sklearn.model_selection import cross_val_predict

fig, ax = plt.subplots()
ax.scatter(y_val, prediction, edgecolors=(0, 0, 0))
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()


# In[60]:



model_score=model.score(x_train, y_train)
model_score

