#!/usr/bin/env python
# coding: utf-8

# Car Price Prediction Model

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[20]:


car_data=pd.read_csv("car data.csv")


# In[21]:


car_data


# In[22]:


car_data.head()


# In[23]:


car_data.info()


# In[13]:


car_data.isnull().sum()


# In[24]:


car_data.describe()


# In[25]:


car_data.columns


# In[27]:


print(car_data['Fuel_Type'].value_counts())


# In[28]:


print(car_data['Selling_Price'].value_counts())
print(car_data['Transmission'].value_counts())


# In[29]:


# visualize the data


# In[30]:


fuel_type=car_data['Fuel_Type']
seller_type=car_data['Seller_Type']
selling_price=car_data['Selling_Price']
transmission_type=car_data['Transmission']


# In[31]:


from matplotlib import style


# In[33]:


style.use('ggplot')
fig = plt.figure(figsize=(15,5))
fig.suptitle('Visualizing categorical data columns')
plt.subplot(1,3,1)
plt.bar(fuel_type,selling_price, color='royalblue')
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.subplot(1,3,2)
plt.bar(seller_type, selling_price, color='red')
plt.xlabel("Seller Type")
plt.subplot(1,3,3)
plt.bar(transmission_type, selling_price, color='purple')
plt.xlabel('Transmission type')
plt.show()


# In[34]:


fig, axes = plt.subplots(1,3,figsize=(15,5), sharey=True)
fig.suptitle('Visualizing categorical columns')
sns.barplot(x=fuel_type, y=selling_price, ax=axes[0])
sns.barplot(x=seller_type, y=selling_price, ax=axes[1])
sns.barplot(x=transmission_type, y=selling_price, ax=axes[2])


# In[39]:


petrol_data=car_data.groupby('Fuel_Type').get_group('Petrol')
petrol_data.describe()


# In[40]:


seller_data=car_data.groupby('Seller_Type').get_group('Dealer')
seller_data.describe()


# In[41]:


#manual encoding
car_data.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)
#one hot encoding
car_data = pd.get_dummies(car_data, columns=['Seller_Type', 'Transmission'], drop_first=True)


# In[42]:


car_data.head()


# In[49]:


plt.figure(figsize=(10,7))
sns.heatmap(car_data.corr(),annot=True)
plt.title('correlation between the columns')
plt.show()


# In[51]:


fig=plt.figure(figsize=(7,5))
plt.title('Correlation between present price and selling price')
sns.regplot(x='Present_Price', y='Selling_Price', data=car_data)


# In[54]:


x=car_data.drop(['Car_Name','Selling_Price'],axis=1)
y=car_data['Selling_Price']


# In[55]:


print("shape of x is",x.shape)
print("shape of y is",y.shape)


# In[63]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[64]:


print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)


# In[65]:


ss=StandardScaler()


# In[67]:


x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[68]:


model=LinearRegression()


# In[75]:


model.fit(x_train,y_train)


# In[80]:


pred=model.predict(x_test)


# In[81]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[82]:


print("MAE: ", (metrics.mean_absolute_error(pred, y_test)))
print("MSE: ", (metrics.mean_squared_error(pred, y_test)))
print("R2 score: ", (metrics.r2_score(pred, y_test)))


# In[84]:


sns.regplot(x=pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("Actual vs predicted price")
plt.show()


# In[ ]:




