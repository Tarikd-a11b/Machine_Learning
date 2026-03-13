import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Load dataset
df = pd.read_csv(f"C:/Users/bilal/Desktop/Datasets/car_price_prediction_.csv")

#Insepct dataset
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())

df=df.rename(columns={"Fuel Type":"FuelType"})

#Drop Unnecessary Columns
df=df.drop(["Car ID","Model"],axis=1)
# print(df.head())

#Count Unique Values
df["Brand"].value_counts()

#OneHot Encoding
df=pd.get_dummies(df,columns=["Brand","FuelType","Transmission"],drop_first=True)
# print(df.head())

#print(df["Condition"].unique())
#Ordinal Encoding
condition_map={"New":3,"Used":2,"Like New":1}
df['Condition']=df['Condition'].map(condition_map)
#print(df.head())

#visualize price distribution
# plt.figure(figsize=(10, 6))
# sns.histplot(df['Price'], kde=True, color='blue')
# plt.title('Fiyat Dağılımı (Price Distribution)')
# plt.show()

# plt.figure(figsize=(12, 10))
# # Sadece sayısal sütunlar arasındaki korelasyonu hesapla
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Özellikler Arası Korelasyon Matrisi')
# plt.show()

# Verinin büyüklüğüne göre sadece temel sütunları seçmek hız kazandırır
# basic_cols = ['Price', 'Year', 'Engine Size', 'Mileage', 'Condition']
# sns.pairplot(df[basic_cols], diag_kind='kde')
# plt.show()

import math
# def plot_all_histograms(df):
#     num_cols = df.select_dtypes(include=[np.number]).columns
#     n_cols = 3
    
    
#     n_rows = math.ceil(len(num_cols) / n_cols)
#     plt.figure(figsize=(5 * n_cols, 4 * n_rows))

#     for i, col in enumerate(num_cols, 1):
#         plt.subplot(n_rows, n_cols, i)
#         sns.histplot(df[col], bins=10,kde=True)
#         plt.title(f'Histogram of {col}')
        
#     plt.tight_layout()
#     plt.show()
# plot_all_histograms(df)

#Model Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

X=df.drop("Price",axis=1)
y=df["Price"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f"MAE: {mae}")
print(f"R2 Score: {r2}")


# "Tüm doğru yöntemleri (Encoding, Scaling, Feature Selection) uygulamamıza rağmen R2 skoru negatif çıkmıştır. 
# Bunun sebebi veri setindeki özellikler (Brand, Year, vb.) ile hedef değişken (Price) arasında mantıksal bir bağ bulunmaması ve 
# verinin rastgele (synthetic/noise) olmasıdır."
