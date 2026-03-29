# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset.
2. Drop unnecessary columns and convert categorical variables using one-hot encoding.
3. Separate the dataset into features (X) and target variable (Y).
4. Standardize the feature and target data using StandardScaler.
5. Split the dataset into training and testing sets.
6. Train the SGD Regressor model using the training data.
7. Predict on test data and evaluate performance using MSE, R², and MAE.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data, drop_first=True)

x=data.drop('price',axis=1)
y=data['price']

Scaler=StandardScaler()
x = Scaler.fit_transform(x)
y = Scaler.fit_transform(np.array(y).reshape(-1,1))

x_train, x_test, y_train, y_test=train_test_split(x, y,test_size=0.2,random_state=42)

sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(x_train,y_train)

y_pred=sgd_model.predict(x_test)

mse=mean_squared_error(y_test,y_pred)
r2score=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

print('Name: a.Jannathul Shaban')
print('Reg.No: 212225220043')
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R2_Score:",r2score)

print("\nModel Coefficients")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

plt.scatter(y_test, y_pred) 
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices using SDG Regressor")
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color='red')
plt.show()



```

## Output:
<img width="420" height="140" alt="image" src="https://github.com/user-attachments/assets/4f4b43b4-82af-4449-a34b-113832e1d69c" />

## Model Coefficients:
<img width="871" height="255" alt="image" src="https://github.com/user-attachments/assets/4e582fd7-85b0-4606-af79-c1a77891a675" />

## Actual VS Predicted Value:
<img width="731" height="582" alt="image" src="https://github.com/user-attachments/assets/a9056f42-729f-43a4-8671-dfaf15db6279" />





## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
