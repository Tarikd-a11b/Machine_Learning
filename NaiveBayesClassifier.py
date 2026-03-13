
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("11-iris.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df["Species"].value_counts())

sns.pairplot(df)
print(plt.show())

sns.scatterplot(x=df["SepalLengthCm"],y=df["SepalWidthCm"],hue=df["Species"])
print(plt.show())

sns.scatterplot(x=df["PetalLengthCm"],y=df["PetalWidthCm"],hue=df["Species"])
print(plt.show())

df=df.drop("Id",axis=1)
from sklearn.preprocessing import LabelEncoder
Labelencoder=LabelEncoder()
df["Species"]=Labelencoder.fit_transform(df["Species"])
print(df.head())
print(df.tail())

X=df.drop("Species",axis=1)
y=df["Species"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train_scaled,y_train)
y_pred=gnb.predict(X_test_scaled)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print("Confusion matris: \n",confusion_matrix(y_pred,y_test))
print("Accuracy score : ",accuracy_score(y_pred,y_test))
print("Classification report:",classification_report(y_pred,y_test))


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_pred_gnb = gnb.predict(X_test_scaled)


log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

svm_model = SVC(kernel='linear') 
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)



models = {
    "Gaussian Naive Bayes": y_pred_gnb,
    "Logistic Regression": y_pred_log,
    "Support Vector Machine": y_pred_svm
}

print("="*50)
for name, pred in models.items():
    print(f"\n MODEL: {name}")
    print(f"Accuracy Score: {accuracy_score(y_test, pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("-" * 30)