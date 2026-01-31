import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#email classification
# subject_formality_score->sender formality_score
# sender_relationship_score->sender relationship score
# email_type->0=personal, 1=work email 

#loading the dataset
df=pd.read_csv("9-email_classification_svm.csv")
print(df.head())


#exploratory data analysis
x=df.drop("email_type", axis=1)
y=df["email_type"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=15)
sns.scatterplot(x=df["subject_formality_score"],y=df["sender_relationship_score"],hue=df["email_type"])
print(plt.show())

# Using Linear Kernel
from sklearn.svm import SVC
svc=SVC(kernel='linear')
svc.fit(x_train, y_train)
y_pred=svc.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Using RBF Kernel
rbf_svc=SVC(kernel='rbf')
rbf_svc.fit(x_train, y_train)
y_rbf_pred=rbf_svc.predict(x_test)
print(confusion_matrix(y_test, y_rbf_pred))
print(classification_report(y_test, y_rbf_pred))

#load lone_risk dataset
loan_df=pd.read_csv("9-loan_risk_svm.csv")
print(loan_df.head())
print(loan_df.info())
print(loan_df.isnull().sum())

sns.scatterplot(x=loan_df["credit_score_fluctuation"], y=loan_df["recent_transaction_volume"], hue=loan_df["loan_risk"])
print(plt.show())

x=loan_df.drop("loan_risk",axis=1)
y=loan_df["loan_risk"]

#linear
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.25,random_state=15)
linear=SVC(kernel="linear")
linear.fit(x_train,y_train)
y_pred3=linear.predict(x_test)
print(classification_report(y_pred3,y_test))
print(confusion_matrix(y_pred3,y_test))
#rbf
rbf=SVC(kernel="rbf")
rbf.fit(x_train,y_train)
y_pred4=rbf.predict(x_test)
print(classification_report(y_pred4,y_test))
print(confusion_matrix(y_pred4,y_test))
#polly
poly=SVC(kernel="poly")
poly.fit(x_train,y_train)
y_pred5=poly.predict(x_test)
print(classification_report(y_pred5,y_test))
print(confusion_matrix(y_pred5,y_test))
#sigmoid
sigmoid=SVC(kernel="sigmoid")
sigmoid.fit(x_train,y_train)
y_pred6=sigmoid.predict(x_test)
print(classification_report(y_pred6,y_test))
print(confusion_matrix(y_pred6,y_test))

#hyperparameter tuning
SVC()
param_grid={
    "C":[0.1,1,10,100,1000],
    "kernel":["rbf"],
    "gamma":["scale","auto"]
}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=SVC(),param_grid=param_grid,cv=5)
grid.fit(x_train,y_train)
print(grid.best_params_)

y_pred7=grid.predict(x_test)
print(classification_report(y_pred7,y_test))
print(confusion_matrix(y_pred7,y_test))

#load dataset seismic activity
df_seismic=pd.read_csv("9-seismic_activity_svm.csv")
print(df_seismic.head())

sns.scatterplot(x=df_seismic["underground_wave_energy"],y=df_seismic["vibration_axis_variation"],hue=df_seismic["seismic_event_detected"])
print(plt.show())

#manual rbf kernel
df_seismic["underground_wave_energy**2"]=df_seismic["underground_wave_energy"]**2
df_seismic["vibration_axis_variation**2"]=df_seismic["vibration_axis_variation"]**2
df_seismic["underground_wave_energy*vibration_axis_variation"]=df_seismic["underground_wave_energy"]*df_seismic["vibration_axis_variation"]
print(df_seismic.head())

x=df_seismic.drop("seismic_event_detected",axis=1)
y=df_seismic["seismic_event_detected"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=15)
print(x_test)

import plotly.express as px
fig=px.scatter_3d(df_seismic,x="underground_wave_energy**2",y="vibration_axis_variation**2",z="underground_wave_energy*vibration_axis_variation",color="seismic_event_detected")
print(fig.show())


linear=SVC(kernel="linear")
linear.fit(x_train,y_train)
y_pred8=linear.predict(x_test)
print(classification_report(y_pred8,y_test))
print(confusion_matrix(y_pred8,y_test))

#auto rbf
df_seismic=pd.read_csv("9-seismic_activity_svm.csv")
x=df_seismic.drop("seismic_event_detected",axis=1)
y=df_seismic["seismic_event_detected"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=15)
rbf=SVC(kernel="rbf")
rbf.fit(x_train,y_train)
y_pred10=rbf.predict(x_test)
print(classification_report(y_pred10,y_test))
print(confusion_matrix(y_pred10,y_test))