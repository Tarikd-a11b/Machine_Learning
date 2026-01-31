
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn as sk

from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline


df=pd.read_csv('3-customersatisfaction.csv')
print(df.head())

df.drop('Unnamed: 0', axis=1, inplace=True)
print(df.head())
print(df.info())

plt.scatter(df['Customer Satisfaction'], df['Incentive'],color='g')
plt.xlabel('Customer Satisfaction')
plt.ylabel('Incentive')
plt.title('Customer Satisfaction vs Incentive')
plt.show()

#dependent and independent variables
X = df[['Customer Satisfaction']]
y = df['Incentive']

#train-test split
x_train,x_test,y_train,y_test=train_test_split(X, y, test_size=0.2,random_state=15)

#scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# --- Linear Regression ---
linearRegression = LinearRegression()
linearRegression.fit(x_train, y_train)

y_pred_linear = linearRegression.predict(x_test)
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_linear))

# --- Polynomial Regression ---
poly = PolynomialFeatures(degree=2) 
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_regression = LinearRegression()

poly_regression.fit(x_train_poly, y_train)

y_pred_poly = poly_regression.predict(x_test_poly)
score = r2_score(y_test, y_pred_poly)
print("Polynomial Regression R2 Score:", score)

# --- Visualization ---
plt.scatter(x_train, y_train, label='Actual Data')
plt.scatter(x_train, poly_regression.predict(x_train_poly), color='r', label='Poly Prediction')
plt.legend()
plt.show()


#degree=3
poly3 = PolynomialFeatures(degree=3)
x_train_poly3 = poly3.fit_transform(x_train)
x_test_poly3 = poly3.transform(x_test)
poly_regression3 = LinearRegression()
poly_regression3.fit(x_train_poly3, y_train)
y_pred_poly3 = poly_regression3.predict(x_test_poly3)
score3 = r2_score(y_test, y_pred_poly3)
print("Polynomial Regression (degree=3) R2 Score:", score3)


#New Data 

new_df=pd.read_csv('3-newdatas.csv')
print(new_df.head())

new_df.rename(columns={"0":"Customer Satisfaction"}, inplace=True)

x_new = new_df[['Customer Satisfaction']]
x_new_scaled = scaler.transform(x_new)
x_new_poly = poly.transform(x_new_scaled)
y_new = poly_regression.predict(x_new_poly)
plt.plot(x_new,y_new,"r",label="New Predictions")
plt.scatter(x_train,y_train,label="Training Points")
plt.legend()
plt.show()

#Pipeline Approach

def poly_regression(degree):
    poly_features = PolynomialFeatures(degree=degree)
    lin_reg = LinearRegression()
    scaler = StandardScaler()
    
  
    pipeline_model = Pipeline([
        ("standard_scaler", scaler),
        ("poly_features", poly_features),
        ("lin_reg", lin_reg)
    ])
    
    pipeline_model.fit(x_train, y_train)
    score = pipeline_model.score(x_test, y_test)
    print(f"Degree {degree} R2 Score: {score}")

    y_pred_new = pipeline_model.predict(x_new)
    
    plt.plot(x_new, y_pred_new, label=f"Degree {degree} Predictions")
    plt.scatter(x_train, y_train, alpha=0.3, label="Training Points")
    plt.legend()
    plt.show()

poly_regression(2)
poly_regression(3)

