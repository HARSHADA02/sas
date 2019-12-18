#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


# In[54]:


data=pd.read_csv(r"C:\nilam\wholesale-customers-data-set\Wholesale customers data.csv")


# In[25]:


data.head()


# In[26]:


data.info()


# In[27]:


data.drop(labels=(['Channel','Region']),axis=1,inplace=True)


# In[30]:


data.head()


# In[55]:


te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
data= pd.DataFrame(te_ary, columns=te.columns_)
data


# In[59]:


from mlxtend.frequent_patterns import apriori

apriori(data, min_support=150)


# In[33]:


data.describe()


# In[35]:


data.info()


# In[38]:


indices = [22,154,398]


# In[39]:


samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
print("Chosen samples of wholesale customers dataset:")
display(samples)


# In[41]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


# In[43]:


new_data = data.drop('Milk',axis=1)


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(new_data, data['Milk'], test_size=0.25, random_state=1)


# In[47]:


Regressor=DecisionTreeRegressor(random_state=1)


# In[49]:


Regressor.fit(X_train,y_train)


# In[51]:


score = Regressor.score(X_test, y_test)
print(score)

