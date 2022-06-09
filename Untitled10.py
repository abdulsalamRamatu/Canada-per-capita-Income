#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[14]:



canada_df=pd.read_csv("C:\\Users\\Ramatu\\.jupyter\\py-master\ML\\1_linear_reg\\Exercise\\canada.csv")
canada_df


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.title("Canada per capital income")

plt.scatter(canada_df.year, canada_df.perCapitaIncome, color='g', marker='+')


# In[18]:


reg= linear_model.LinearRegression()
reg.fit(canada_df[['year']],canada_df.perCapitaIncome)


# In[19]:


reg.predict([[2020]])


# In[ ]:




