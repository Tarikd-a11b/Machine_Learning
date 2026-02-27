import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=sns.load_dataset("titanic")

# print(df.head())
# print(df.info())
# print(df.describe())

#eda

# sns.barplot(data=df,x='sex',y="survived")
# plt.show()

# print(df["sex"].value_counts())

# sns.catplot(data=df,x="pclass",hue="survived",kind="count")
# plt.show()

# print(df["survived"].value_counts())

#print(df.groupby("pclass")["survived"].mean()) # sınıflara göre hayatta kalma oranı

# sns.histplot(data=df,x="age",kde=True)
# plt.show()

# sns.countplot(data=df,x="who",hue="survived")
# plt.show()

#featuring engineering

# print(df.isnull().sum())
# print(df.shape)

df=df.drop(["deck","embark_town","alive"],axis=1)
#print(df.isnull().sum())

df["age"]=df["age"].fillna(df["age"].mean())
df["embarked"]=df["embarked"].fillna(df["embarked"].mode()[0])

# print(df.isnull().sum())

df["adult_male"]=df["adult_male"].astype(int)
df["alone"]=df["alone"].astype(int)

X=df.drop("survived",axis=1)
y=df["survived"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_cols=["sex","embarked","class","who"]
preprocessor=ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),categorical_cols)
    ],remainder="passthrough"
)

X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)

encoded_cols=preprocessor.get_feature_names_out()
X_train=pd.DataFrame(X_train,columns=encoded_cols)
X_test=pd.DataFrame(X_test,columns=encoded_cols)

import lightgbm as lgb
from sklearn.metrics import classification_report,confusion_matrix

lgbm=lgb.LGBMClassifier(verbosity=-1)
lgbm.fit(X_train,y_train)

y_pred=lgbm.predict(X_test)

# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
 
importances=lgbm.feature_importances_
feature_names=X_train.columns

feature_importances=pd.DataFrame({"feature":feature_names,"importance":importances})
feature_importances=feature_importances.sort_values(by="importance",ascending=False)
# print(feature_importances)

#hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV
lgb_model=lgb.LGBMClassifier(random_state=15)

param_dist={
    "n_estimators":[100,200,300,400,500],
    "learning_rate":[0.01,0.05,0.1,0.2,0.3],
    "num_leaves":[31,63,127,255],
    "max_depth":[-1,10,20,30,40,50], #-1 =none
    "min_child_samples":[20,40,60,80,100],
    "subsample":[0.6,0.7,0.8,0.9,1.0],
    "colsample_bytree":[0.6,0.7,0.8,0.9,1.0]
    }
random_search=RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring="accuracy",
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train,y_train)

# print("Best parameters: ",random_search.best_params_)

y_pred=random_search.predict(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


from xgboost import XGBClassifier
xgb=XGBClassifier(n_estimators=100)
xgb.fit(X_train,y_train)
y_pred=xgb.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

params={
    "n_estimators":[100,200,300,400,500],
    "learning_rate":[0.01,0.05,0.1,0.2,0.3],
    "max_depth":[-1,10,20,30,40,50], #-1 =none
    "colsample_bytree":[0.6,0.7,0.8,0.9,1.0]
}

random_search=RandomizedSearchCV(estimator=XGBClassifier(),param_distributions=params,cv=5,scoring="accuracy",n_jobs=-1)
random_search.fit(X_train,y_train)
y_pred=random_search.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))





