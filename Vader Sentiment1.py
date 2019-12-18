#!/usr/bin/env python
# coding: utf-8

# In[5]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[6]:


analyzer=SentimentIntensityAnalyzer()


# In[11]:


def test(sentence):
    score=analyzer.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


# In[12]:


test("I am itrested in data Science")


# In[13]:


test('I like mango')


# In[14]:


test('i love india')


# In[18]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

analyzer = SentimentIntensityAnalyzer()
translator = Translator()

def sentiment_analyzer_scores(text):
    trans = translator.translate(text).text

    score = analyzer.polarity_scores(trans)
    lb = score['compound']
    if lb >= 0.05:
        return 'positive'
    elif (lb > -0.05) and (lb < 0.05):
        return 'neutral'
    else:
        return 'negative'
      
print(sentiment_analyzer_scores('programmieren ist lustig'))   

print(sentiment_analyzer_scores('service is rude'))

print(sentiment_analyzer_scores('food is delitious'))
# In[21]:


print(sentiment_analyzer_scores('I  love mango'))


# In[17]:


get_ipython().system(' pip install googletrans')


# In[29]:


from textblob  import TextBlob


# In[23]:


get_ipython().system(' pip install Textblob')


# In[30]:


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


# In[31]:


blob=TextBlob(text)


# In[38]:


blob


# In[37]:


blob.tags


# In[35]:


import nltk


# In[36]:


nltk.download('averaged_perceptron_tagger')
 


# In[40]:


blob.sentences


# In[41]:


for sentence in blob.sentences:
    print(sentence.sentiment.polarity)


# In[ ]:





