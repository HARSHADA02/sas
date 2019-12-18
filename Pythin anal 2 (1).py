#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd


# In[77]:


import numpy as np


# In[78]:


st=pd.read_csv("E:\\akshay\\datasets\\starbucks_drinkMenu_expanded.csv")


# In[79]:


st.head()


# In[80]:


st.dropna(inplace=True)


# In[81]:


st


#How to check datatype
print(ba.dtypes)
# In[82]:


st['Beverage_prep'].value_counts()


# In[83]:


st1=st.groupby('Beverage_category', as_index=False)['Caffee'].mean()

st1



# In[84]:

# how to change column names

t.rename(columns={'Unnamed: 13':'Target'}, inplace=True)

#How to remove columns

sa.drop(columns=['Unnamed: 0', 'chas'])

# How to takeprediction in dataframe
pd=(pd.DataFrame(prediction))
pd


# In[85]:


#How to remove Last three characters from text

st2= st1['Beverage_category'].map(lambda x: str(x)[:-6])


# In[86]:


st2


# In[87]:

# Remove First three characters
st5 = st['Beverage_category'].str[3:]


# In[88]:


print(st5)


#Remove last 1 number
st2=st['Carbohydrates'].map(lambda x: str(x)[:-1])





#Remove first 1 number

st5=st['Carbohydrates'].map(lambda x: str(x)[1:])


# In[91]:

#How to add two datasets
df =pd.concat([st, st5], axis=1)


# In[92]:


df.head()


# In[ ]:

# How to remove first three numbers 

st['Caffee'] = st['Caffee'].apply(str).str.replace('.', ',')

st5 = st['Caffee'].str[4:]



#  How to remove last two number
st2=st['Caffee'].map(lambda x: str(x)[:-2])



#how to take first three characters

st6=st['Beverage_prep'].astype(str).str[0:3]


# In[32]:


st6


# In[38]:


#How to take last two characters

st['Beverage_prep'].str[-2:]






#How to take last 10 rows

df2[-10:]



#Convert A String Categorical Variable To A Numeric Variable
def score_to_numeric(x):
    if x=='Yes':
        return 1
    if x=='No':
        return 0	
z['Calories1'] = z['Calories'].apply(score_to_numeric)


#Convert A String Categorical Variable To A Numeric Variable
def score_to_numeric(x):
    if x < 100:
        return "C"
    if x < 250:
        return "B"
    if x < 350:
        return "A"
st['Carbohydrates'] = st['Carbohydrates'].apply(score_to_numeric)



link-
https://chrisalbon.com/python/data_wrangling/convert_categorical_to_numeric/
#  Convert A Number Categorical Variable To A Numeric Variable
# '<' not supported between instances of 'str' and 'int'

 
def myFunction(myData):
    col = [[0,0],[0,0],[0,0]]
    for i in range(len(tableData)):
        for word in tableData[i]:
            if len(word) > col[i][1]:
                col[i][1]=word
        print(col[i][1]) 

def score_to_numeric(x):
    if x<17:
        return 0
    if x>=17:
        return 1
adv['Sales'] = adv['Sales'].apply(score_to_numeric)


# Value Nan . too long , too large , float(32)



X_train = X_train.fillna(X_train.mean())


# In[45]:


y_train = y_train.fillna(y_train.mean())


# In[50]:


X_test = X_test.fillna(X_test.mean())


# In[51]:


y_test = y_test.fillna(y_test.mean())



# "[["nflation_annual_cpi","exch_usd" ] not in index"


columns=["nflation_annual_cpi","exch_usd"]

cr= cr.reindex(columns=columns)#cr is dataset


# How to save dataset on Excel Sheet:


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('C:\\Users\\akshay\\Desktop\\retv.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
rev1.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


#return codecs.utf_8_decode(input, errors, True) UnicodeDecodeError:
 'utf8' codec can't decode byte 0xe9 in position 10: invalid continuation byte

ef=pd.read_csv('E:\\data science project\\economic\\economic_freedom.csv'  ,encoding='latin-1')

#For specific rows:
ef2=ef.iloc[[176, 177], :]

#For specific Columns
ef1=ef.iloc[:, [24, 25]]

#select two columns
le1.loc[(le1['CountryDisplay']=="India") & (le1['GhoDisplay']=="Healthy life expectancy (HALE) at birth (years)")]


