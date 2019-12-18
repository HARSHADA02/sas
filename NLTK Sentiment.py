#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 


# In[2]:


def word_feats(words):
    return dict([(word, True) for word in words])
 


# In[3]:


positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]


# In[4]:


positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
 


# In[5]:


train_set = negative_features + positive_features + neutral_features
 


# In[12]:


import pandas as pd


# In[13]:


sent=pd.read_csv('E:\\data science project\\zomato.csv')


# In[14]:


sent.head()


# In[6]:


classifier = NaiveBayesClassifier.train(train_set) 
 


# In[7]:


# Predict
neg = 0
pos = 0
sentence = "Awesome movie, I liked it"
sentence = sentence.lower()
words = sentence.split(' ')


# In[8]:


for word in words:
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1


# In[9]:


print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))


# In[ ]:




