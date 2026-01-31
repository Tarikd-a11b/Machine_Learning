
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data=pd.read_csv('Summary of Weather.csv')
print(data.head())
# Check for missing values
print(data.isnull().sum())
data=data.drop(columns=['WindGustSpd','PoorWeather','DR','SPD','SND','FT','FB','FTI','ITH','PGT','TSHDSBRSGF','SD3','RHX','RHN','RVG','WTE','SNF'])
print(data.info())

print(data.isnull().sum())
print(data[data.isnull().any(axis=1)])


print(data['Snowfall'].unique())
print(data['PRCP'].unique())

data['Snowfall'] = data['Snowfall'].fillna(0)
data['PRCP'] = data['PRCP'].fillna(0)

def temizle(deger):
    d = str(deger).strip()
    
    if d == 'T':
        return 0.0  
    elif d == '#VALUE!':
        return 0.0  
    else:
        try:
            return float(d) 
        except ValueError:
            return 0.0 

data['Snowfall'] = data['Snowfall'].apply(temizle)
data['PRCP'] = data['PRCP'].apply(temizle)
print("------------------------------------------------------------------------------------------")
print(data['Snowfall'].unique())
print(data['PRCP'].unique())
print(data.head())
print(data.tail())


# Fill missing values with mean of the columns
for col in ['MAX', 'MIN', 'MEA']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data['MAX'] = data['MAX'].fillna(data['MAX'].median())
data['MIN'] = data['MIN'].fillna(data['MIN'].median())
data['MEA'] = data['MEA'].fillna(data['MEA'].mean())

data['Snowfall'] = data['Snowfall'].fillna(0)
data['PRCP'] = data['PRCP'].fillna(0)

print("--- Temizlik Sonrası Kontrol ---")
print(data.isnull().sum())
print(data.dtypes)

#visualize the dataset
# Sadece sayısal (int ve float) sütunları seç
numeric_data = data.select_dtypes(include=[np.number])
# Korelasyon matrisini hesapla
corr_matrix = numeric_data.corr()
# Isı haritasını (Heatmap) çizdir
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Sayısal Değişkenler Arası Korelasyon Matrisi")
plt.show()

# Define features and target variable
X = data.drop('MEA', axis=1)
y = data['MEA']
print(X.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

#dependent and independent features
df=data.copy()
print(df.head())
sns.heatmap(df.corr())
print(plt.show())
