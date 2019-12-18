#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[6]:



st=pd.read_csv(r"Downloads\retv3.csv")
st.head()


# In[7]:


st['Revenue']=st['Total_Collection']-st['Total_Sales']


# In[8]:


st.head()


# In[10]:


#st.iloc[:10, [0,8]]


# In[11]:


st1=st.loc[st['Revenue'] > 0]


# In[12]:


st1.head()


# In[15]:


st1.sort_values('Revenue', axis=0, inplace=True, na_position='last',ascending=False)


# In[16]:


st1


# In[23]:


st2=st1.iloc[:5,:]


# In[27]:


st2


# In[24]:


import matplotlib.pyplot as plt


# In[26]:


plt.scatter(x='Center', y='Revenue', data=st2)


# In[33]:


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('C:\\Users\\Administrator\\Desktop\\Report.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
st2.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


# In[34]:


st4=st1.groupby('Region', as_index=False)['Revenue'].mean()


# In[35]:


st4


# In[36]:


st1.dropna(inplace=True)


# In[37]:


print(st.dtypes)


# In[38]:


st['Region'].value_counts()


# In[40]:



#How to remove columns

st2.drop(columns=['ARPC'])


# In[41]:


#How to add two datasets
df =pd.concat([st2, st4], axis=1)


# In[42]:


df


# In[43]:


#How to take last 10 rows

st[-10:]



# In[ ]:




