import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("10-diamonds.csv")
print(df.head())

df = df.drop(["Unnamed: 0"], axis=1)
df = df.drop(df[df["x"]==0].index)
df = df.drop(df[df["y"]==0].index)
df = df.drop(df[df["z"]==0].index)

df = df[(df["depth"]<75)&(df["depth"]>45)]
df = df[(df["table"]<80)&(df["table"]>40)]
df = df[(df["y"]<30)]
df = df[(df["z"]<30)&(df["z"]>2)]

X= df.drop(["price"],axis =1)
y= df["price"]
print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=15)
from sklearn.preprocessing import LabelEncoder

# previously we just encoded the columns with label encoder like this and it worked
# however if we want to save the encoders we should have seperated them
# so i will use new encoding
#label_encoder = LabelEncoder()
#for col in ['cut', 'color', 'clarity']:
#    X_train[col] = label_encoder.fit_transform(X_train[col])
#    X_test[col] = label_encoder.transform(X_test[col])

encoders = {}
for col in ['cut', 'color', 'clarity']:
    encoders[col] = LabelEncoder()
    X_train[col] = encoders[col].fit_transform(X_train[col])
    X_test[col] = encoders[col].transform(X_test[col])
print(X_train.head())

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.svm import SVR
svr=SVR(C=1000, gamma=0.1, kernel='rbf')
# we get this from the hyperparameter tuning that we have done before
# refer: https://github.com/atilsamancioglu/MachineLearningNotebooks/blob/main/10-SVMRegressor.ipynb

from sklearn.metrics import r2_score
svr.fit(X_train_scaled, y_train)
y_pred=svr.predict(X_test_scaled)
score=r2_score(y_test,y_pred)
print("R2 Score", score)

import pickle
with open('30-diamond_model_complete.pkl', 'wb') as f:
    pickle.dump({
        'model': svr,
        'encoders': encoders,
        'scaler': scaler
    }, f)
pd.DataFrame(X_test_scaled).to_csv("30_testdatascaled.csv", index=False)