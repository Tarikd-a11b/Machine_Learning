
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df=pd.read_csv('4-Algerian_forest_fires_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Display dataset information
print(df.info())
print(df.isnull().sum())

#show the NaN values in the dataset
print(df[df.isnull().any(axis=1)])

# Drop the row with index 122 which contains NaN values
df.drop(122, inplace=True)

#add new column Region.and set its values based on the index
df.loc[:123,"Region"]=0
df.loc[123:,"Region"]=1
print(df.head())
print(df.tail())

#drop all NaN values and reset the index(row 123 and 168 contain NaN values)
df=df.dropna().reset_index(drop=True)
print(df.isnull().sum())

#remove space characeters from column names
df.columns=df.columns.str.strip() 
print(df.columns)

#check unique values in the "day" column
print(df["day"].unique())
print(df[df["day"]=="day"])

df.drop(122, inplace=True)

# Convert "day" column values from string to integer
df[["day","month","year","Temperature","RH","Ws"]]=df[["day","month","year","Temperature","RH","Ws"]].astype(int)
print(df.info())
# Convert other relevant columns to float
df[["Rain","FFMC","DMC","DC","ISI","BUI","FWI"]]=df[["Rain","FFMC","DMC","DC","ISI","BUI","FWI"]].astype(float)
print(df.info())
# Encode the target variable "Classes"
df['Classes']=np.where(df['Classes'].str.contains('not fire'),0,1)
print(df['Classes'].value_counts())

# Visualize the correlation matrix
sns.heatmap(df.corr())
plt.show()

# Drop unnecessary columns
df.drop(['day','month','year'],axis=1,inplace=True)
print(df.info())

#dependent and independent features
x=df.drop("FWI",axis=1)
y=df["FWI"]
print(x.head())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=15)

#redudancy,multicollineartiy,overfitting
def correlation_for_dropping(df, threshold):
    col_corr = set()  
    corr_matrix = df.corr()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return list(col_corr) 

columns_to_drop = correlation_for_dropping(x_train, 0.85)
print(f"Silinecek Sütunlar: {columns_to_drop}")
x_train.drop(columns=columns_to_drop, axis=1, inplace=True)
x_test.drop(columns=columns_to_drop, axis=1, inplace=True)
print(x_train.shape)
print(x_test.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=x_train)
plt.title("Before Scaling")
plt.subplot(1,2,2)
sns.boxplot(data=x_train_scaled)
plt.title("After Scaling")
print(plt.show())


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

linear=LinearRegression()
linear.fit(x_train_scaled,y_train)
y_pred=linear.predict(x_test_scaled)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"Linear Regression - MSE: {mse}, MAE: {mae}, R2: {r2}")
plt.scatter(y_test,y_pred)
print(plt.show())

from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(x_train_scaled,y_train)
y_pred_lasso=lasso.predict(x_test_scaled)
mse_lasso=mean_squared_error(y_test,y_pred_lasso)
mae_lasso=mean_absolute_error(y_test,y_pred_lasso)
r2_lasso=r2_score(y_test,y_pred_lasso)
print(f"Lasso Regression - MSE: {mse_lasso}, MAE: {mae_lasso}, R2: {r2_lasso}")
plt.scatter(y_test,y_pred_lasso)
print(plt.show())

from sklearn.linear_model import Ridge
ridge=Ridge()
ridge.fit(x_train_scaled,y_train)
y_pred_ridge=ridge.predict(x_test_scaled)
mse_ridge=mean_squared_error(y_test,y_pred_ridge)
mae_ridge=mean_absolute_error(y_test,y_pred_ridge)
r2_ridge=r2_score(y_test,y_pred_ridge)
print(f"Ridge Regression - MSE: {mse_ridge}, MAE: {mae_ridge}, R2: {r2_ridge}")
plt.scatter(y_test,y_pred_ridge)
print(plt.show())

from sklearn.linear_model import ElasticNet
elasticnet=ElasticNet()
elasticnet.fit(x_train_scaled,y_train)
y_pred_en=elasticnet.predict(x_test_scaled)
mse_en=mean_squared_error(y_test,y_pred_en)
mae_en=mean_absolute_error(y_test,y_pred_en)
r2_en=r2_score(y_test,y_pred_en)
print(f"ElasticNet Regression - MSE: {mse_en}, MAE: {mae_en}, R2: {r2_en}")
plt.scatter(y_test,y_pred_en)
print(plt.show())

#lasso croos validation
from sklearn.linear_model import LassoCV
lasso_cv=LassoCV(cv=5)
lasso_cv.fit(x_train_scaled,y_train)
y_pred_lasso_cv=lasso_cv.predict(x_test_scaled)
mse_lasso_cv=mean_squared_error(y_test,y_pred_lasso_cv)
mae_lasso_cv=mean_absolute_error(y_test,y_pred_lasso_cv)
r2_lasso_cv=r2_score(y_test,y_pred_lasso_cv)
print(f"Lasso CV Regression - MSE: {mse_lasso_cv}, MAE: {mae_lasso_cv}, R2: {r2_lasso_cv}")
plt.scatter(y_test,y_pred_lasso_cv)
print(plt.show())

#ridge cross validation
from sklearn.linear_model import RidgeCV
ridge_cv=RidgeCV(cv=5)
ridge_cv.fit(x_train_scaled,y_train)
y_pred_ridge_cv=ridge_cv.predict(x_test_scaled)
mse_ridge_cv=mean_squared_error(y_test,y_pred_ridge_cv)
mae_ridge_cv=mean_absolute_error(y_test,y_pred_ridge_cv)
r2_ridge_cv=r2_score(y_test,y_pred_ridge_cv)
print(f"Ridge CV Regression - MSE: {mse_ridge_cv}, MAE: {mae_ridge_cv}, R2: {r2_ridge_cv}")
plt.scatter(y_test,y_pred_ridge_cv)
print(plt.show())

#elasticnet cross validation
from sklearn.linear_model import ElasticNetCV
elasticnet_cv=ElasticNetCV(cv=5)
elasticnet_cv.fit(x_train_scaled,y_train)
y_pred_en_cv=elasticnet_cv.predict(x_test_scaled)
mse_en_cv=mean_squared_error(y_test,y_pred_en_cv)
mae_en_cv=mean_absolute_error(y_test,y_pred_en_cv)
r2_en_cv=r2_score(y_test,y_pred_en_cv)
print(f"ElasticNet CV Regression - MSE: {mse_en_cv}, MAE: {mae_en_cv}, R2: {r2_en_cv}")
plt.scatter(y_test,y_pred_en_cv)
print(plt.show())

#lazzy fitt
import lazypredict
from lazypredict.Supervised import LazyRegressor
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import Numpy as np

diabetes= datasets.load_diabetes()
X, y = shuffle(diabetes.data, diabetes.target, random_state=13)

offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

print(X_train.shape)
print(X_test.shape)
print(X[0])
print(y[0])

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)


