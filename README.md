# Food-Delivery-Time-prediction
In this project, we present the results of an in-depth analysis of ride-sharing data, focusing on estimating
delivery time. The analysis includes Exploratory Data Analysis (EDA) as well as the implementation of 
various regression models such as Linear Regression, Decision Tree Regressor, XGB Regressor, and a 
neural network LSTM model. The evaluation metrics used to assess the model performance are Mean 
Squared Error (MSE).
The dataset used in this analysis comprises ride-sharing data that includes information about geographic 
location coordinates (latitude and longitude), different other features . Prior to analysis, the dataset 
underwent data preprocessing, including handling missing values, removing outliers, and normalizing the 
feature.
Several regression models were implemented to predict riding distances based on geographic 
coordinates and other features.
Linear Regression:
MSE: 1.89
The Linear Regression model yielded a relatively low MSE, suggesting a reasonable fit to the data. This 
indicates that a linear relationship between geographic coordinates and riding distances exists.
XGB Regressor:
MSE: 5.36
The XGB Regressor exhibited a higher MSE compared to Linear Regression, which indicates that the 
model's predictions may have been less accurate due to the complexity of the underlying relationship.
LSTM Neural Network:
MSE: 0.0575
The LSTM model, a type of recurrent neural network (RNN), demonstrated the best performance among 
the models. Its significantly low MSE suggests that the model effectively captured temporal 
dependencies in the data, resulting in accurate predictions.
Conclusion:
In this analysis, we explored ride-sharing data using EDA techniques and predicted riding distances using 
various regression models. The results indicated that the LSTM neural network outperformed other 
models in terms of accuracy, as evidenced by its lowest MSE value. This suggests that incorporating 
temporal information through the LSTM architecture was beneficial in improving the predictive power of 
the model. The findings of this study can have practical implications for improving ride-sharing services 
by enhancing distance estimation accuracy, leading to better customer experiences and operational 
efficiency. Further research could focus on fine-tuning model parameters and exploring additional 
features to potentially improve the performance of the predictive models
