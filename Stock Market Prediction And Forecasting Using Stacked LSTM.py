#!/usr/bin/env python
# coding: utf-8

# # LGM DATASCIENCE INTERNSHIP | AUG2021 
# 
# Level: Beginner
# 
# Task 3: Stock Market Prediction And Forecasting Using Stacked LSTM

# # Import the librariesJust like any other Python program we first import all the necessary libraries such as NumPy, Pandas, SciKitlearn, MatPlotLib, and Keras. 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler


# In[4]:


df=pd.read_csv("https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv")
df.head()


# # DATA EXPLORATION

# In[5]:


df.head()


# In[6]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.dropna(inplace = True, how = 'all')


# In[10]:


df.isnull().sum()


# In[11]:


len(df)


# In[12]:


df.isna().any()


# In[14]:


df.describe()


# In[15]:


price_mean=df['Close'].mean()
price_mean


# # Plot close data

# In[24]:


df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']
plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')


# # Sort date and close

# In[47]:


data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]


# In[29]:


data.ewm(span=200).mean()['Close'].plot(figsize=(15,15),label='200EMA')
data.rolling(window=200).mean()['Close'].plot(figsize=(15,15),label='200SMA')
data['Close'].plot(label='Close')
plt.legend()
plt.ylabel('price')
plt.show()


# In[31]:


training_orig = data.loc[:,['Close']]
training_orig


# In[32]:


training_orig['Close'].plot


# In[41]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(data['Close'])
plt.xlabel('Data',fontsize=18)
plt.ylabel('Close Price USD($)',fontsize=18)
plt.show()


# # Datatype conversion

# In[54]:


df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']


# In[55]:


data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])


# In[56]:


for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]


# In[57]:


# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

# shapes of training set
print('\n Shape of training set:')
print(train.shape)

# shapes of validation set
print('\n Shape of validation set:')
print(valid.shape)


# In[58]:


# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# making predictions
preds = []
for i in range(0,valid.shape[0]):
    a = train['Close'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)

# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms)


# In[59]:


#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])


# In[60]:


#setting index as date values
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]


# In[61]:


#create features
from fastai.structured import  add_datepart
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp


# In[62]:


#split into train and validation
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[ ]:


#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
rms


# In[ ]:




