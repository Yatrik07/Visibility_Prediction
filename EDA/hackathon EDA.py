#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np


# In[4]:


data = pd.read_csv("InputFile.csv")


# In[5]:


data


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.describe()


# In[23]:


data.isna().sum()


# In[9]:


data.shape


# In[10]:


data.columns


# In[11]:


data.nunique()


# In[14]:


data.isnull().sum()


# In[24]:


corelation = data.corr(method = 'spearman')


# In[25]:


plt.figure(figsize = (10,10) , dpi = 200)
sbn.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)


# In[26]:


corelation = data.corr()


# In[27]:


plt.figure(figsize = (10,10) , dpi = 200)
sbn.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot=True)


# ## Here StationPressure and SeaLevelPressure are having 100% correlartion

# ## Here DewPointTempF and WERBULBTEMPF are having 97% correlation

# ## So We will remove Multicollinearity

# In[17]:


sbn.pairplot(data)


# In[31]:


import warnings
warnings.filterwarnings('ignore')
for i in list(data.columns)[1:]:
    sns.distplot(data[i])
    plt.show()


# ## Here not all features are Normally Distributed so we will apply scaling on data

# In[ ]:




