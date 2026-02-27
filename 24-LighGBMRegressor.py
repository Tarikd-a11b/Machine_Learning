import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('24-medical_cost.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# print(df["sex"].value_counts())
# print(df["smoker"].value_counts())
# print(df["region"].value_counts())

#sns.lineplot(x="age",y="charges",hue="sex",data=df,errorbar=None)
#plt.show()

# sns.histplot(data=df,x="charges",kde=True)
# plt.show()

#Feature Engineering

df=df.drop("Id",axis=1)
#print(df.head())

df["sex"]=df["sex"].map({"male":0,"female":1})
df["smoker"]=df["smoker"].map({"no":0,"yes":1})
#print(df.value_counts())

#one hot encoding ->region
X=df.drop("charges",axis=1)
y=df["charges"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_cols=["region"]

preprocessor=ColumnTransformer(transformers=
                               [('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical_cols)],remainder='passthrough')

X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)

from lightgbm import LGBMRegressor
model=LGBMRegressor()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error
# print("R2 Score:",r2_score(y_pred,y_test))
# print("MSE:",mean_squared_error(y_pred,y_test))

#Hyperparameter Tuning

param_grid={
    'num_leaves':[31,50,100],
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':[100,200,500],
    'max_depth':[3,5,7],
    'min_child_samples':[20,50,100],
    'subsample':[0.6,0.8,1.0],
    'colsample_bytree':[0.6,0.8,1.0],
    'reg_alpha':[0,0.1,0.5,1],
    'reg_lambda':[0,0.1,0.5,1]
}

from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

random_search=RandomizedSearchCV(
    estimator=LGBMRegressor(verbosity=-1),
    param_distributions=param_grid,
    cv=5,
    verbose=0,
    random_state=15,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

random_search.fit(X_train,y_train)
best_params=random_search.best_params_
#print("Best Hyperparameters:",best_params)

y_pred=random_search.predict(X_test)
# print("R2 Score:",r2_score(y_pred,y_test))
# print("MSE:",mean_squared_error(y_pred,y_test))

#Transformation

from scipy.stats import boxcox
y_trained_transformed,lambda_y=boxcox(y_train)
model=LGBMRegressor()
model.fit(X_train,y_trained_transformed)

y_pred_transformed=model.predict(X_test)

#inverse Box-Cox transformation

def inverse_boxcox(y,lambda_):
    if lambda_==0:
        return np.exp(y)
    else:
        return np.power(y*lambda_+1,1/lambda_)
y_pred_original=inverse_boxcox(y_pred_transformed,lambda_y)
print("R2 Score:",r2_score(y_pred_original,y_test))
print("MSE:",mean_squared_error(y_pred_original,y_test))
