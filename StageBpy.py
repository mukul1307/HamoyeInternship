#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('energydata_complete.csv')


# In[3]:


data.head()


# In[13]:


df_1 = data.drop(columns = 'date')


# In[14]:


df_1


# In[26]:


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
df_1 = pd.DataFrame(scalar.fit_transform(df_1),columns = df_1.columns)
feature_var1 = df_1['T2']
target_var1 = df_1['T6']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature_var1, target_var1, test_size=0.10, random_state=1)

x_train = x_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x_train, y_train)
predicted_values = model1.predict(x_test)


# In[11]:


from sklearn.metrics import r2_score
r2score = r2_score(y_test, predicted_values)
round(r2score, 2)


# In[12]:


df_2 = data.drop(columns = ['date', 'lights'])


# In[15]:


df_2


# In[27]:


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()

df_2 = pd.DataFrame(scalar.fit_transform(df_2), columns=df_2.columns)

feature_var_df = df_2.drop(columns='Appliances')
target_var = df_2['Appliances']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature_var_df, target_var, test_size=0.30, random_state=42)

from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(x_train, y_train)

predicted_values = model2.predict(x_test)


# In[17]:


from sklearn.metrics import mean_absolute_error
mean_abs_error = mean_absolute_error(y_test, predicted_values)
round(mean_abs_error, 2)


# In[22]:


res_sum_sq = np.sum(np.square(y_test - predicted_values))
round(res_sum_sq, 2)


# In[21]:


from sklearn.metrics import mean_squared_error
rt_mean_sq_error = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rt_mean_sq_error, 3)


# In[23]:


from sklearn.metrics import r2_score
coeff_det = r2_score(y_test, predicted_values)
round(coeff_det, 2)


# In[33]:


def obtain_weights(model, feature, column):
    wts = pd.Series(model.coef_, feature.columns).sort_values()
    wts_df = pd.DataFrame(wts).reset_index()
    wts_df.columns = ['Features',column]
    wts_df[column].round(3)
    return wts_df


# In[52]:


obtained_weights = obtain_weights(model2, x_train, 'Weights-Linear Model')

obtained_weights


# In[37]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_2 = pd.DataFrame(scaler.fit_transform(df_2), columns=df_2.columns)

feature_var2_df = df_2.drop(columns='Appliances')
target_var = df_2['Appliances']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature_var2_df, target_var, test_size=0.70, random_state=42)

from sklearn.linear_model import Ridge
rr_model = Ridge(alpha=0.4)
rr_model.fit(x_train, y_train)

predicted_values = rr_model.predict(x_test)


# In[38]:


from sklearn.metrics import mean_squared_error
rt_mean_sq_error = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rt_mean_sq_error, 3)


# In[39]:


df_3 = data.drop(columns = ['date', 'lights'])


# In[41]:


df_3


# In[43]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_3 = pd.DataFrame(scaler.fit_transform(df_3), columns=df_3.columns)

feature_var3_df = df_3.drop(columns='Appliances')
target_var = df_3['Appliances']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature_var3_df, target_var, test_size=0.30, random_state=42)

from sklearn.linear_model import Lasso
lr_model = Lasso(alpha=0.001)
lr_model.fit(x_train, y_train)

predicted_values = lr_model.predict(x_test)


# In[44]:


def obtain_weights_2(model, feature, column):
    wts = pd.Series(model.coef_, feature.columns).sort_values()
    wts_df = pd.DataFrame(wts).reset_index()
    wts_df.columns = ['Features', column]
    wts_df[column].round(3)
    return wts_df


# In[51]:


obtained_lasso_weights = obtain_weights_2(lr_model, x_train, 'Weights-Lasso Regression Model')

obtained_lasso_weights


# In[53]:


from sklearn.metrics import mean_squared_error
rt_mean_sq_error = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rt_mean_sq_error, 3)

