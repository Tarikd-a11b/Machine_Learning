import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

data=load_breast_cancer(as_frame=True)
df=data.frame
#print(df.head())

X=df.drop(columns='target')
y=df['target']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)



from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
columns=data.feature_names
X_train=pd.DataFrame(X_train,columns=columns)
X_test=pd.DataFrame(X_test,columns=columns)

#Before PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

logistic=LogisticRegression()
gbc=GradientBoostingClassifier()

logistic.fit(X_train,y_train)
gbc.fit(X_train,y_train)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# print("Logistic Regression")
y_pred_logistic=logistic.predict(X_test)
# print(accuracy_score(y_test,y_pred_logistic))
# print(classification_report(y_test,y_pred_logistic))
# print(confusion_matrix(y_test,y_pred_logistic))

# print("-----------------------")
# print("Gradient Boosting Classifier")
y_pred_gbc=gbc.predict(X_test)
# print(accuracy_score(y_test,y_pred_gbc))
# print(classification_report(y_test,y_pred_gbc))
# print(confusion_matrix(y_test,y_pred_gbc))



#After PCA
from sklearn.decomposition import PCA

pca=PCA(n_components=4)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

X_train_pca=pd.DataFrame(X_train_pca,columns=["PC 1","PC 2","PC 3","PC 4"])
X_test_pca=pd.DataFrame(X_test_pca,columns=["PC 1","PC 2","PC 3","PC 4"])

logistic.fit(X_train_pca,y_train)
gbc.fit(X_train_pca,y_train)

# print("Logistic Regression")
y_pred_logistic=logistic.predict(X_test_pca)
# print(accuracy_score(y_test,y_pred_logistic))
# print(classification_report(y_test,y_pred_logistic))
# print(confusion_matrix(y_test,y_pred_logistic))

# print("-----------------------")
# print("Gradient Boosting Classifier")
y_pred_gbc=gbc.predict(X_test_pca)
# print(accuracy_score(y_test,y_pred_gbc))
# print(classification_report(y_test,y_pred_gbc))
# print(confusion_matrix(y_test,y_pred_gbc))


