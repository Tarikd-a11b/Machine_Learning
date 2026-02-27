
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("21-housing.csv")
# print(df.head())
# print(df.info())
# print(df.describe())

# sns.lineplot(data=df,x="housing_median_age",y="median_house_value")
# plt.title("House price by age")
# plt.show()

# print(df.isnull().sum())

# print(df.columns)

columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value']

fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(15,12))
fig.suptitle("Distributtion",fontsize=18,fontweight="bold")

for i,col in enumerate(columns):
   row=i//3
   col_idx=i%3
   ax=axes[row,col_idx]
   sns.histplot(data=df,x=col,kde=True,ax=ax,bins=30)
   ax.set_title(col,fontsize=10,fontstyle="italic")

# plt.tight_layout()
# plt.show()


#outliers detect function

def find_outliers_iqr(df, threshold=1.5):
    """
    Veri çerçevesindeki sayısal sütunlar için IQR yöntemiyle outlier analizi yapar.
    """
    outliers_summary = {}
    
    # Sadece sayısal sütunları seçiyoruz
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)

        # Filtreleme
        outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        total_count = len(df)

        outliers_summary[col] = {
            "count": outliers_count,
            "percentage": round((outliers_count / total_count) * 100, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        }

    return outliers_summary

outliers = find_outliers_iqr(df)

# Sözlüğü DataFrame'e çevirip transpoze (.T) alıyoruz
outlier_df = pd.DataFrame(outliers).T
# print(outlier_df)

def remove_outliers_from_column(df, target_col, threshold=1.5):
    # 'col' yerine 'target_col' kullanmalıyız
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR  # Burası + olmalı
    
    return df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]

def remove_outliers_from_all_columns(df, threshold=1.5):
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR 
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
    return df_clean
# print("original data shape:",df.shape)
df_target_clean=remove_outliers_from_column(df,"median_house_value")
# print("only target column cleaning shape:",df_target_clean.shape)
# df_all_clean=remove_outliers_from_all_columns(df)
# print("all columns cleaning shape:",df_all_clean.shape)


df_target_clean["total_bedrooms"]=df_target_clean["total_bedrooms"].fillna(df_target_clean["total_bedrooms"].median())
# print(df_target_clean.isnull().sum())

# print(df_target_clean["ocean_proximity"].value_counts())

df_target_clean=pd.get_dummies(df_target_clean,columns=["ocean_proximity"],drop_first=True)
#print(df_target_clean.head())


X=df_target_clean.drop("median_house_value",axis=1)
y=df_target_clean["median_house_value"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3,random_state=15)

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def evaluate_model(true,predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mse)
    r2_square=r2_score(true,predicted)
    return mae,mse,rmse,r2_square

#evaluate models

models={
    "Linear Regression":LinearRegression(),
    "Ridge":Ridge(),
    "Lasso":Lasso(),
    "KNeighborsRegressor":KNeighborsRegressor(),
    "DecisionTreeRegressor":DecisionTreeRegressor(),
    "RandomForestRegressor":RandomForestRegressor(),
    "AdaBoostRegressor":AdaBoostRegressor(),
    "GradientBoostingRegressor":GradientBoostingRegressor(),
    "XGBRegressor":XGBRegressor()
}

for i in range (len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)
    
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)

  
    model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    # print(list(models.keys())[i])
    # print("Model performance for traning set")
    # print("-"*30)
    # print("MAE",model_train_mae)
    # print("MSE",model_train_mse)
    # print("RMSE",model_train_rmse)
    # print("R2",model_train_r2)

    # print("\n")
    # print("Model performance for test set")
    # print("-"*30)
    # print("MAE",model_test_mae)
    # print("MSE",model_test_mse)
    # print("RMSE",model_test_rmse)
    # print("R2",model_test_r2)
    # print("-"*30)
    # print("\n")

#hyperparameter tuning
xgboost_params={
    "learning_rate":[0.1,0.01],
    "max_depth":[5,8,12,20,30],
    "n_estimators":[100,200,300,500],
    "colsample_bytree":[0.3,0.4,0.5,0.7,1]
}

from sklearn.model_selection import RandomizedSearchCV

randomized_cv=RandomizedSearchCV(estimator=XGBRegressor(),param_distributions=xgboost_params,cv=5,n_jobs=-1)
randomized_cv.fit(X_train,y_train)

# print(randomized_cv.best_params_)

#best model:{'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 1}

model=XGBRegressor(n_estimators=200,max_depth=5,learning_rate=0.1,colsample_bytree=1)
model.fit(X_train,y_train)

y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

  
model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

print("XGBoostRegressor")
print("Model performance for traning set")
print("-"*30)
print("MAE",model_train_mae)
print("MSE",model_train_mse)
print("RMSE",model_train_rmse)
print("R2",model_train_r2)
print("\n")
print("Model performance for test set")
print("-"*30)
print("MAE",model_test_mae)
print("MSE",model_test_mse)
print("RMSE",model_test_rmse)
print("R2",model_test_r2)
print("-"*30)
print("\n")