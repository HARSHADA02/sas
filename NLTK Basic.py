#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk


# In[4]:


from nltk.tokenize import sent_tokenize, word_tokenize


# In[5]:


text = '''
The titular threat of The Blob has always struck me as the ultimate movie
monster: an insatiably hungry, amoeba-like mass able to penetrate
virtually any safeguard, capable of--as a doomed doctor chillingly
describes it--"assimilating flesh on contact.
Snide comparisons to gelatin be damned, it's a concept with the most
devastating of potential consequences, not unlike the grey goo scenario
proposed by technological theorists fearful of
artificial intelligence run rampant.
'''


# In[6]:


print(word_tokenize(text))


# In[7]:


print(sent_tokenize(text))


# In[8]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
 

stopWords = set(stopwords.words('english'))
words = word_tokenize(text)
wordsFiltered = []
 
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
 
print(wordsFiltered)


# In[9]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
 
ps = PorterStemmer()
 

words = word_tokenize(text)
 
for word in words:
    print(word + ":" + ps.stem(word))


# In[ ]:




