#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[71]:


from sklearn.datasets import load_iris


# In[72]:


data


# In[73]:


dataset=load_iris()


# In[74]:


dataset


# In[75]:


print(dataset.DESCR)


# In[76]:


x=dataset.data


# In[77]:


y=dataset.target


# In[78]:


y


# In[79]:


x


# In[80]:


plt.plot(x[:,0][y==0]*x[0:,1][y==0],x[:,1][y==0]*x[:,2][y==0],'r.',label='Setosa')
plt.plot(x[:,0][y==1]*x[0:,1][y==1],x[:,1][y==1]*x[:,2][y==1],'b.',label='Versicolor')
plt.plot(x[:,0][y==2]*x[0:,1][y==2],x[:,1][y==2]*x[:,2][y==2],'g.',label='Verginica')
plt.legend()
plt.show()


# In[81]:


from sklearn.preprocessing import StandardScaler


# In[82]:


ss=StandardScaler()


# In[83]:


x=ss.fit_transform(x)


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[86]:


from sklearn.linear_model import LogisticRegression


# In[91]:


log_reg=LogisticRegression()


# In[92]:


log_reg.fit(x_train,y_train)


# In[93]:


log_reg.score(x_test,y_test)


# In[94]:


log_reg.score(x,y)


# In[ ]:




