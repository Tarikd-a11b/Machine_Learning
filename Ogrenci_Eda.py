import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk


df=pd.read_csv(r"C:\Users\bilal\Desktop\Python\2-multiplegradesdataset.csv")
print(df.head())
print(df.describe())
print(df.isnull().sum())

sns.pairplot(df)
print(plt.show())

print(df.corr())

#draw best fit line
sns.regplot(x=df['Study Hours'], y=df['Exam Score'])
print(plt.show())

#create heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
print(plt.show())

#independent and dependent variables
X = df[['Study Hours',"Sleep Hours","Attendance Rate","Social Media Hours"]]
y = df['Exam Score']

#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)

from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(X_train,y_train)
print("Model trained successfully.")
print(LinearRegression())


new_student=[[5,7,90,2]]
new_student_scaled=scaler.transform(new_student)
regression.predict(new_student_scaled)
print("Predicted Exam Score for new student:", regression.predict(new_student_scaled))


#prediction
y_pred=regression.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

#adjusted score
print("Adjusted R^2 Score:", 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1))

#visualize actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores")
print(plt.show())


regression.coef_
regression.intercept_
print("Coefficients:", regression.coef_)
print("Intercept:", regression.intercept_)

student=[
    [4,6,85,3],
    [6,8,65,1],
    [3,5,80,4]
]
student_scaled=scaler.transform(student)
predictions=regression.predict(student_scaled)
for i, pred in enumerate(predictions):
    print(f"Predicted Exam Score for student {i+1}: {pred}")