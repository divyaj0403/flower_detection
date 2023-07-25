#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd


# In[14]:


from pandas import DataFrame


# In[15]:


import numpy as np


# In[16]:


import seaborn as sns


# In[17]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


IrisDF =pd.read_csv("iris-data.csv")


# In[19]:


IrisDF.head()


# In[20]:


IrisDF.info()


# In[21]:


IrisDF


# In[22]:


IrisDF.describe()


# In[23]:


IrisDF.columns


# In[27]:


x=IrisDF.drop(columns=["class"],axis=1)
x


# In[28]:


sns.pairplot(IrisDF)


# In[29]:


sns.heatmap(x.corr(),annot=True)


# In[55]:


X=IrisDF[['sepal-length', 'sepal-width', ' petal-width']]

y=IrisDF['petal-length']


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.40, random_state=101)


# In[58]:


from sklearn.linear_model import LinearRegression


# In[59]:


lm = LinearRegression()


# In[60]:


lm.fit (X_train, y_train)


# In[61]:


coeff_df = pd.DataFrame (lm.coef_, X.columns, columns=['Coefficient'])


# In[62]:


coeff_df


# In[63]:


predictions = lm.predict (X_test)


# In[64]:


plt.scatter (y_test, predictions)


# In[65]:


sns.distplot((y_test-predictions),bins=50);


# In[ ]:




