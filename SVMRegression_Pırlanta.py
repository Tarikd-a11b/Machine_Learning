
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("10-diamonds.csv")
print(df.head())
print(df.shape)

df=df.drop("Unnamed: 0" ,axis=1)
print(df.head())
print(df.describe())
print(df.isnull().sum())

#x,y ve z değerleri bu pırlantanın boyutunu belirityor ama bazı satırlarda bu değerler 0 ve bu mümkün olamayacağı için ya silmemiz gerekli yada uygun değerleri atamalıyız bundan ötürü sayılarını bir kontorl ediyoruz.
print(len(df[df["x"]==0]),len(df[df["y"]==0]),len(df[df["z"]==0]))

df[df["x"]==0].index
df[df["y"]==0].index
df[df["z"]==0].index

#değerleri siliyoruz
df=df.drop(df[df["x"]==0].index)
df=df.drop(df[df["y"]==0].index)
df=df.drop(df[df["z"]==0].index)
print(df.describe())
print(df.shape)

sns.pairplot(df)
print(plt.show())

sns.scatterplot(x=df["x"],y=df["price"])
print(plt.show())
sns.scatterplot(x=df["y"],y=df["price"])
print(plt.show())
sns.scatterplot(x=df["z"],y=df["price"])
print(plt.show())

print(len(df[(df["depth"] < 75) & (df["depth"] > 45)]))
print(len(df[(df["table"] < 75) & (df["table"] > 40)]))
print(len(df[(df["z"] < 30) & (df["z"] > 2)]))
print(len(df[df["y"] < 20]))


df = df[(df["depth"] < 75) & (df["depth"] > 45)]
df = df[(df["table"] < 75) & (df["table"] > 40)]
df = df[(df["z"] < 30) & (df["z"] > 2)]
df = df[df["y"] < 20]

print(df.describe())


print(df["cut"].value_counts())
print(df["color"].value_counts())
print(df["clarity"].value_counts())

#Encoding
X=df.drop("price",axis=1)
y=df["price"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
for col in ["cut","color","clarity"]:
    X_train[col]=label_encoder.fit_transform(X_train[col])
    X_test[col]=label_encoder.transform(X_test[col])

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_sclaed=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
linear=LinearRegression()
linear.fit(X_train_sclaed,y_train)
y_pred=linear.predict(X_test_scaled)
mae=mean_absolute_error(y_pred,y_test)
mse=mean_squared_error(y_pred,y_test)
score=r2_score(y_pred,y_test)
print("Mean absolute error:",mae)
print("Mean squared error:",mse)
print("R2 score:",score)
plt.scatter(y_test,y_pred)
print(plt.show())

from  sklearn.svm import SVR
svr=SVR()
svr.fit(X_train_sclaed,y_train)
y_pred=svr.predict(X_test_scaled)
mae=mean_absolute_error(y_pred,y_test)
mse=mean_squared_error(y_pred,y_test)
score=r2_score(y_pred,y_test)
print("Mean absolute error:",mae)
print("Mean squared error:",mse)
print("R2 score:",score)
plt.scatter(y_test,y_pred)
print(plt.show())

#hyperparameter
from sklearn.model_selection import GridSearchCV
param_grid={
    "C":[0.1,1,10,100,1000],
    "gamma":[1,0.1,0.001],
    "kernel":["rbf","linear"]
}
grid=GridSearchCV(estimator=SVR(),param_grid=param_grid,n_jobs=-1,verbose=3)
grid.fit(X_train_sclaed,y_train)
y_pred=grid.predict(X_test_scaled)
print(grid.best_params_)
mae=mean_absolute_error(y_pred,y_test)
mse=mean_squared_error(y_pred,y_test)
score=r2_score(y_pred,y_test)
print("Mean absolute error:",mae)
print("Mean squared error:",mse)
print("R2 score:",score)
plt.scatter(y_test,y_pred)
print(plt.show())