import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
data=pd.read_csv('7-cyber_attack_data.csv')
print(data.head())

# Preprocess data
x=data.drop("attack_type",axis=1)
y=data["attack_type"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=15)
from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
model=LogisticRegression()
model.fit(x_test,y_test)
y_pred=model.predict(x_test)
print("Predicted values are:" ,y_pred)

# Evaluate model
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
score=accuracy_score(y_pred,y_test)
print("Score:",score)
print(classification_report(y_pred,y_test))
print("Confusion Matrix is: \n",confusion_matrix(y_pred,y_test))

#Hyperparameter Tuning
model=LogisticRegression()
penalty=['l1','l2','elasticnet']
c_values=[100,10,1.0,0.1]
solver=['lbfgs','liblinear','saga','newton-cg','sag','newton-cholesky']
params=dict(penalty=penalty,C=c_values,solver=solver)
#grid search cv
from sklearn.model_selection import GridSearchCV,StratifiedKFold
cv=StratifiedKFold()
grid=GridSearchCV(estimator=model,param_grid=params,cv=cv,scoring='accuracy',n_jobs=-1)
grid.fit(x_train,y_train)
print(grid.best_params_)
print(grid.best_score_)
y_pred=grid.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)
print(" Accuracy is:",accuracy)
print(classification_report(y_pred,y_test))
print(" Confusion Matrix is: \n",confusion_matrix(y_pred,y_test))
#random search cv
from sklearn.model_selection import RandomizedSearchCV
model=LogisticRegression()
randomcv=RandomizedSearchCV(estimator=model,param_distributions=params,cv=5,n_iter=10,scoring="accuracy")
randomcv.fit(x_train,y_train)
print(randomcv.best_params_)
print(randomcv.best_score_)
y_pred=randomcv.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)
print(" Accuracy is:",accuracy)
print(classification_report(y_pred,y_test))
print(" Confusion Matrix is: \n",confusion_matrix(y_pred,y_test))
randomcv.fit(x_train,y_train)
print(randomcv.best_params_)
print("Random Search Best Score:",randomcv.best_score_)

# onevs rest classifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
onevsonemodel=OneVsOneClassifier(LogisticRegression())
onevsrestmodel=OneVsRestClassifier(LogisticRegression())
onevsonemodel.fit(x_train,y_train)
y_pred=onevsonemodel.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)
print(" One vs One Classifier Accuracy is:",accuracy)
print(classification_report(y_pred,y_test))
print(" Confusion Matrix is: \n",confusion_matrix(y_pred,y_test))
onevsrestmodel.fit(x_train,y_train)
y_pred=onevsrestmodel.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)
print(" One vs Rest Classifier Accuracy is:",accuracy)
print(classification_report(y_pred,y_test))
print(" Confusion Matrix is: \n",confusion_matrix(y_pred,y_test))

