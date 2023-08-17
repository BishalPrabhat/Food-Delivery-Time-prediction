#!/usr/bin/env python
# coding: utf-8

# # Food Delivery Time Prediction using Machine Learning Algorithms

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


pip install plotly


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df = pd.read_csv("food_delivery.csv", encoding='latin1')


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


df.duplicated().sum()


# In[10]:


df.isnull().sum()


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.nunique()


# In[14]:


object_columns = df.select_dtypes(include='object').columns
print("Object Columns:")
print(object_columns)
print()

numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print("Numerical Columns:")
print(numerical_columns)


# In[15]:


df['Type_of_order'].unique()


# In[16]:


df['Type_of_order'].value_counts()


# In[17]:


plt.figure(figsize=(15, 6))
counts = df['Type_of_order'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('hls'))
plt.title('Type_of_order')
plt.show()


# In[18]:


import plotly.graph_objects as go


# In[19]:


fig = go.Figure(data=[go.Bar(x=df['Type_of_order'].value_counts().index, y=df['Type_of_order'].value_counts())])
fig.update_layout(
        title= 'Type_of_order',
        xaxis_title="Categories",
        yaxis_title="Count"
    )
fig.show()


# In[20]:


counts = df['Type_of_order'].value_counts()
fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
fig.update_layout(title= 'Type_of_order')
fig.show()


# In[21]:


df['Type_of_vehicle'].unique()


# In[22]:


df['Type_of_vehicle'].value_counts()


# In[23]:


plt.figure(figsize=(15, 6))
counts = df['Type_of_vehicle'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette('hls'))
plt.title('Type_of_vehicle')
plt.show()


# In[24]:


fig = go.Figure(data=[go.Bar(x=df['Type_of_vehicle'].value_counts().index, y=df['Type_of_vehicle'].value_counts())])
fig.update_layout(
        title= 'Type_of_vehicle',
        xaxis_title="Categories",
        yaxis_title="Count"
    )
fig.show()


# In[25]:


counts = df['Type_of_vehicle'].value_counts()
fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts)])
fig.update_layout(title= 'Type_of_vehicle')
fig.show()


# In[26]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.histplot(df[i], kde = True, bins = 20, palette = 'hls')
    plt.xticks(rotation = 90)
    plt.show()


# In[27]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.distplot(df[i], kde = True, bins = 20)
    plt.xticks(rotation = 90)
    plt.show()


# In[28]:


import plotly.express as px

for column in numerical_columns:
    fig = px.histogram(df, x=column, nbins=20, histnorm='probability density')
    fig.update_layout(title=f"Histogram of {column}", xaxis_title=column, yaxis_title="Probability Density")
    fig.show()


# In[29]:


for column in numerical_columns:
    fig = px.box(df, y=column)
    fig.update_layout(title=f"Box Plot of {column}", yaxis_title=column)
    fig.show()


# In[30]:


for column in numerical_columns:
    fig = px.violin(df, y=column)
    fig.update_layout(title=f"Violin Plot of {column}", yaxis_title=column)
    fig.show()


# In[31]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.barplot(x = df['Type_of_order'], y = df[i], data = df, ci = None, palette = 'hls')
    plt.show()


# In[32]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.barplot(x = df['Type_of_vehicle'], y = df[i], data = df, ci = None, palette = 'hls')
    plt.show()


# In[33]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.boxplot(x = df['Type_of_order'], y = df[i], data = df, palette = 'hls')
    plt.show()


# In[34]:


for i in numerical_columns:
    plt.figure(figsize=(15,6))
    sns.violinplot(x = df['Type_of_order'], y = df[i], data = df, palette = 'hls')
    plt.show()


# In[35]:


for i in numerical_columns:
    for j in numerical_columns:
        if i != j:
            plt.figure(figsize=(15,6))
            sns.lineplot(x = df[j], y = df[i], data = df, ci = None, palette = 'hls')
            plt.xticks(rotation = 90)
            plt.show()


# In[36]:


df


# In[37]:


# Extracting Time Components
df['hour_of_day'] = pd.to_datetime(df['Time_taken(min)'], unit='m').dt.hour
df['day_of_week'] = pd.to_datetime(df['Time_taken(min)'], unit='m').dt.dayofweek
df['month_of_year'] = pd.to_datetime(df['Time_taken(min)'], unit='m').dt.month


# In[39]:


import math


# In[40]:


# Function to calculate distance between two sets of latitude and longitude coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula to calculate distance
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c

    return distance

# Calculate distance and create the distance feature
df['distance'] = df.apply(lambda row: calculate_distance(row['Restaurant_latitude'], row['Restaurant_longitude'],
                                                        row['Delivery_location_latitude'], row['Delivery_location_longitude']), axis=1)


# In[41]:


# Categorizing Age
age_bins = [0, 30, 50, float('inf')]
age_labels = ['young', 'middle-aged', 'senior']
df['age_category'] = pd.cut(df['Delivery_person_Age'], bins=age_bins, labels=age_labels)


# In[42]:


# Aggregating Ratings
df['avg_ratings'] = df.groupby('Delivery_person_ID')['Delivery_person_Ratings'].transform('mean')


# In[43]:


# Binary Encoding
df = pd.get_dummies(df, columns=['Type_of_order', 'Type_of_vehicle'],drop_first=False,dtype=int)


# In[44]:


# Interaction Features
df['time_ratings_interaction'] = df['Time_taken(min)'] * df['Delivery_person_Ratings']


# In[45]:


#df


# In[46]:


columns_to_drop = ['ID', 'Delivery_person_ID', 'Restaurant_latitude', 'Restaurant_longitude', 
                   'Delivery_location_latitude', 'Delivery_location_longitude']


# In[47]:


# Drop the columns from the dataset
df = df.drop(columns=columns_to_drop)


# In[48]:


df.dtypes


# In[49]:


df.info()


# In[50]:


df.columns


# In[54]:


pip install -U scikit-learn


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[52]:


features_to_scale = ['Delivery_person_Age', 'Delivery_person_Ratings', 'time_ratings_interaction']
features_not_to_scale = ['Time_taken(min)', 'hour_of_day', 'day_of_week', 'month_of_year', 'distance',
                         'age_category', 'avg_ratings', 'Type_of_order_Buffet ', 'Type_of_order_Drinks ',
                         'Type_of_order_Meal ', 'Type_of_order_Snack ', 'Type_of_vehicle_bicycle ',
                         'Type_of_vehicle_electric_scooter ', 'Type_of_vehicle_motorcycle ',
                         'Type_of_vehicle_scooter ']
target = 'Time_taken(min)'


# In[53]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features_to_scale + features_not_to_scale], df[target], test_size=0.25, random_state=42)


# In[54]:


# Perform feature scaling for the appropriate features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[features_to_scale] = scaler.fit_transform(X_train_scaled[features_to_scale])
X_test_scaled[features_to_scale] = scaler.transform(X_test_scaled[features_to_scale])


# In[55]:


# Perform one-hot encoding for the 'age_category' feature
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), ['age_category'])],
    remainder='passthrough'
)
X_train_scaled = ct.fit_transform(X_train_scaled)
X_test_scaled = ct.transform(X_test_scaled)


# In[56]:


# Create and train the linear regression model
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)


# In[57]:


# Make predictions on the test set
y_pred = model_lr.predict(X_test_scaled)


# In[58]:


# Evaluate the model using root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root Mean Squared Error:', rmse)


# In[59]:


from sklearn.metrics import r2_score


# In[60]:


# Calculate R-squared score
r2_lr = r2_score(y_test, y_pred)
print('R-squared Score:', r2_lr)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


# In[61]:


pip install xgboost


# In[62]:


from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from xgboost import XGBRegressor


# In[63]:


# Create and train the Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train_scaled, y_train)


# In[64]:


# Make predictions on the test set using Decision Tree Regressor
y_pred_dt = dt_regressor.predict(X_test_scaled)


# In[65]:


# Calculate R-squared score for Decision Tree Regressor
r2_dt = r2_score(y_test, y_pred_dt)
print('Decision Tree Regressor - R-squared Score:', r2_dt)


# In[66]:


# Calculate Mean Squared Error (MSE) for Decision Tree Regressor
mse_dt = mean_squared_error(y_test, y_pred_dt)
print('Decision Tree Regressor - Mean Squared Error:', mse_dt)


# In[67]:


# Create and train the XGBoost Regressor
xgb_regressor = XGBRegressor(random_state=42)
xgb_regressor.fit(X_train_scaled, y_train)


# In[68]:


# Make predictions on the test set using XGBoost Regressor
y_pred_xgb = xgb_regressor.predict(X_test_scaled)


# In[69]:


# Calculate R-squared score for XGBoost Regressor
r2_xgb = r2_score(y_test, y_pred_xgb)
print('XGBoost Regressor - R-squared Score:', r2_xgb)

# Calculate Mean Squared Error (MSE) for XGBoost Regressor
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print('XGBoost Regressor - Mean Squared Error:', mse_xgb)


# In[74]:


X_train


# In[78]:


import tensorflow as tf


# In[79]:


from tensorflow import keras


# In[80]:


from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[81]:


model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_train_scaled.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()


# In[83]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, batch_size=1, epochs=9)


# In[ ]:




