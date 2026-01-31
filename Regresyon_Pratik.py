
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split


# Load dataset
data = pd.read_csv(r"C:\Users\bilal\Desktop\Python\MachineLearningNotebooks-main\1-studyhours.csv")
print(data.head())

plt.scatter(data['Study Hours'], data['Exam Score'])
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Study Hours vs Exam Score')
plt.show()

X = data[['Study Hours']]
y = data['Exam Score']

type(X)
type(y)

#Test-Train Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(X_train.head())
print(y_train.head())


#standardization
from sklearn.preprocessing import StandardScaler
