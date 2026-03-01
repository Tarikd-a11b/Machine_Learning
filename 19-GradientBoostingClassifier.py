
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('19-heart.csv')

# print("Head:\n",df.head())
# print("\nInfo:\n",df.info())
# print("\nDescribe:\n",df.describe())
# print("\nNull:\n",df.isnull().sum())
# print("\nDuplicated:\n",df.duplicated().sum())

df.hist(bins=40,figsize=(15,10))
#plt.show()

X=df.drop('target',axis=1)
y=df['target']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=15)

def correlation_for_dropping(df,threshold):
   columns_to_drop=set()
   corr=df.corr()
   for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i,j])>threshold:
                col_name=corr.columns[i]
                columns_to_drop.add(col_name)
   return list(columns_to_drop)

correlation_for_dropping(df,0.80)
set()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV

gbc=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
gbc.fit(X_train,y_train)
y_pred=gbc.predict(X_test)

print("Classification Report:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))


#hyperparemeter tuning

parameters={
    'n_estimators':[100,200,300],
    'learning_rate':[0.1,0.01,0.001],
    'max_depth':[3,4,5],
    'loss':['log_loss','exponential'],
    'subsample':[0.8,1]
}

grid_search=GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=parameters,cv=5,n_jobs=-1,verbose=1)
grid_search.fit(X_train,y_train)

print("Best Parameters:",grid_search.best_params_)
print("Best Accuracy:",grid_search.best_score_)

y_pred=grid_search.predict(X_test)

print("Classification Report:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
