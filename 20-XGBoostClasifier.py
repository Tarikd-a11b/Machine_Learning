import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("20-digitalskysurvey.csv")

# print(df.head())
columns_to_drop=["objid","specobjid","rerun","camcol","field","run"]
df.drop(columns_to_drop,axis=1,inplace=True)
# print(df.head())
# print(df.info())
# print(df.describe())

#print(df['class'].value_counts())

# sns.scatterplot(data=df,x="redshift",y="ra",hue="class")
# print(plt.show())

# sns.scatterplot(data=df,x="redshift",y="plate",hue="class")
# print(plt.show())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["class"]=le.fit_transform(df["class"])
# print(df.head())


# sns.pairplot(df,hue="class")
# plt.show()
#the best classification metrics=redshift

# fig, axes=plt.subplots(nrows=1,ncols=3,figsize=(16,4))
# ax=sns.histplot(df[df["class"]==2].redshift,ax=axes[0])
# ax.set_title("Star")

# ax=sns.histplot(df[df["class"]==0].redshift,ax=axes[1])
# ax.set_title("Galaxy")

# ax=sns.histplot(df[df["class"]==1].redshift,ax=axes[2])
# ax.set_title("QSO")

# print(plt.show())


X=df.drop("class",axis=1)
y=df["class"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=15)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=100)
xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

# print("confusion matrix: \n",confusion_matrix(y_pred,y_test))
# print("accuracy score: ",accuracy_score(y_pred,y_test))
# print(classification_report(y_pred,y_test))

params={
    "n_estimators":[100,200,300,500,550],
    "learning_rate":[0.01,0.1,3],
    "max_depth":[5,8,12,20,30,35],
    "colsample_bytree":[0.3,0.4,0.5,0.8,0.10]
}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=XGBClassifier(),param_grid=params,cv=5,n_jobs=-1)
grid.fit(X_train,y_train)

print(grid.best_params_)
print("confusion matrix: \n",confusion_matrix(y_pred,y_test))
print("accuracy score: ",accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))