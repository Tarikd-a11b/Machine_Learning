import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('18-concrete_data.csv')
# print(df.head())
# print(df.columns)
# print(df.isnull().sum())
# print(df.info())
# print(df.describe())


#replace space with underscore
# df.columns=df.columns.str.replace(' ','_')
# print(df.head())

#correlation
# print(df.corr())

#correlation matrix
# sns.heatmap(df.corr())
# plt.show()

# sns.lineplot(data=df,x='Age',y='Strength')
# plt.show()
#Age means sum of days


X=df.drop('Strength',axis=1)
y=df['Strength']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=15)

from sklearn.tree import DecisionTreeRegressor
#first weak learner
tree_reg1=DecisionTreeRegressor(max_depth=3)
tree_reg1.fit(X_train,y_train)
y1=y_train-tree_reg1.predict(X_train)
# print(y1[:5])
#second weak learner
tree_reg2=DecisionTreeRegressor(max_depth=4)
tree_reg2.fit(X_train,y1)
y2=y1-tree_reg2.predict(X_train)
# print(y2[:5])
#third weak learner
tree_reg3=DecisionTreeRegressor(max_depth=4)
tree_reg3.fit(X_train,y2)
y3=y2-tree_reg3.predict(X_train)
# print(y3[:5])

y_pred=sum(tree.predict(X_test) for tree in (tree_reg1,tree_reg2,tree_reg3))
# print(y_pred)

from sklearn.metrics import r2_score
# print(r2_score(y_test,y_pred))



from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(n_estimators=3,max_depth=3,learning_rate=0.1)
gbr.fit(X_train,y_train)

y_pred=gbr.predict(X_test)
print(r2_score(y_test,y_pred))


gbr=GradientBoostingRegressor(n_estimators=100,max_depth=3,learning_rate=0.1)
gbr.fit(X_train,y_train)

y_pred=gbr.predict(X_test)
print(r2_score(y_test,y_pred))

#n_estimator'ın default değeri 100'dür

#hyperparemeter_tuning

params={
    "n_estimators":[100,120,150,200],
    "max_depth":[3,4,5,6,7],
    "learning_rate":[0.01,0.1,0.5,1],
    "loss":["squared_error","absolute_error","huber","quantile"]
}

from sklearn.model_selection import RandomizedSearchCV
rscv=RandomizedSearchCV(estimator=GradientBoostingRegressor(),param_distributions=params,cv=5)
rscv.fit(X_train,y_train)
# print(rscv.best_params_)

gbr=GradientBoostingRegressor(n_estimators=rscv.best_params_['n_estimators'],max_depth=rscv.best_params_['max_depth'],learning_rate=rscv.best_params_['learning_rate'],loss=rscv.best_params_['loss'])
gbr.fit(X_train,y_train)
y_pred=gbr.predict(X_test)
print(r2_score(y_test,y_pred))