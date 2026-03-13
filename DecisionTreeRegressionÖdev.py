import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Date time year-month-day hour:minute:second
# Appliances, energy use in Wh
# lights, energy use of light fixtures in the house in Wh
# T1, Temperature in kitchen area, in Celsius
# RH_1, Humidity in kitchen area, in %
# T2, Temperature in living room area, in Celsius
# RH_2, Humidity in living room area, in %
# T3, Temperature in laundry room area
# RH_3, Humidity in laundry room area, in %
# T4, Temperature in office room, in Celsius
# RH_4, Humidity in office room, in %
# T5, Temperature in bathroom, in Celsius
# RH_5, Humidity in bathroom, in %
# T6, Temperature outside the building (north side), in Celsius
# RH_6, Humidity outside the building (north side), in %
# T7, Temperature in ironing room , in Celsius
# RH_7, Humidity in ironing room, in %
# T8, Temperature in teenager room 2, in Celsius
# RH_8, Humidity in teenager room 2, in %
# T9, Temperature in parents room, in Celsius
# RH_9, Humidity in parents room, in %
# To, Temperature outside (from Chievres weather station), in Celsius
# Pressure (from Chievres weather station), in mm Hg
# RH_out, Humidity outside (from Chievres weather station), in %
# Wind speed (from Chievres weather station), in m/s
# Visibility (from Chievres weather station), in km
# Tdewpoint (from Chievres weather station), Â°C
# rv1, Random variable 1, nondimensional
# rv2, Random variable 2, nondimensional

# Where indicated, hourly data (then interpolated) from the nearest airport weather station (Chievres Airport, Belgium) 
# was downloaded from a public data set from Reliable Prognosis, rp5.ru. Permission was obtained from Reliable Prognosis
#  for the distribution of the 4.5 months of weather data.




#Load dataset
df=pd.read_csv("KAG_energydata_complete.csv")

#Inspect to data
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())


# print(df["lights"].value_counts())
# print(df["Windspeed"].value_counts())

# Date kolonunu datetime tipine çevirdik
df['date'] = pd.to_datetime(df['date'])

# Saat, haftanın günü ve hafta sonu olup olmadığını belirleyen sütunları ekledik
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek # 0=Pazartesi, 6=Pazar
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
# print(df.head())


# Sadece sayısal kolonları seçelim
numeric_df = df.select_dtypes(include=[np.number])

# Korelasyon matrisini hesaplayalım
corr_matrix = numeric_df.corr()

# Görselleştirelim (Heatmap)
# plt.figure(figsize=(16, 10))
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
# plt.title("Korelasyon Matrisi - Enerji Veri Seti")
# plt.show()

# 1. Tamamen gereksiz rv1 ve rv2'yi atalım
df = df.drop(['rv1', 'rv2'], axis=1)

# 2. Birbirinin kopyası olan sıcaklıklardan bazılarını eleyelim
# T6 ve T_out çok benzer, T6'yı atalım
df = df.drop(['T6'], axis=1)

# 3. İç ortam sıcaklıklarını tek bir özellik haline getirebiliriz (Boyut Azaltma)
temp_cols = ['T1', 'T2', 'T3', 'T4', 'T5', 'T7', 'T8', 'T9']
df['T_inside_avg'] = df[temp_cols].mean(axis=1)
df = df.drop(temp_cols, axis=1)

# 4. Aynı işlemi Nem (RH) için de yapabiliriz (Opsiyonel ama önerilir)
rh_cols = ['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_5', 'RH_7', 'RH_8', 'RH_9']
df['RH_inside_avg'] = df[rh_cols].mean(axis=1)
df = df.drop(rh_cols, axis=1)


# Korelasyon matrisi
# plt.figure(figsize=(12, 10))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Korelasyon Matrisi")
# plt.show()

# 'Appliances' kolonunun 10 dakika önceki (bir önceki satır) değerini alalım
df['lag_1'] = df['Appliances'].shift(1)

# İlk satır NaN olacağı için onu silelim veya bir değerle dolduralım
# Genelde silmek en sağlıklısıdır çünkü model NaN değerlerle çalışmaz.
df.dropna(inplace=True)

#X ve y'yi tanımlayalım
X = df.drop(['Appliances', 'date'], axis=1)
y = df['Appliances']

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Decision Tree Regressor modelini oluşturalım
decision_tree = DecisionTreeRegressor(random_state=42)

#Modeli eğitelim
decision_tree.fit(X_train, y_train)

#Test seti üzerinde tahmin yapalım
y_pred = decision_tree.predict(X_test)

#Model performansını değerlendirelim
# print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))

# Sınırlandırılmış bir ağaç deneyelim
dt_optimized = DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42)
dt_optimized.fit(X_train, y_train)
# Yeni skorları kontrol et...

train_score = decision_tree.score(X_train, y_train)
# print(f"Eğitim Seti R2 Skoru: {train_score}")


from sklearn.ensemble import RandomForestRegressor

# --- 1. Yaklaşım: Karar Ağacını Budama (Pruning) ---
# max_depth (derinlik) ve min_samples_leaf (yapraktaki min. örnek) kısıtlaması getiriyoruz
dt_optimized = DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42)
dt_optimized.fit(X_train, y_train)

# Optimize edilmiş ağaç tahminleri
y_pred_opt = dt_optimized.predict(X_test)

# print("\n--- Optimize Edilmiş Karar Ağacı Sonuçları (max_depth=10) ---")
# print(f"Eğitim R2 Skoru: {dt_optimized.score(X_train, y_train):.4f}")
# print(f"Test R2 Skoru: {r2_score(y_test, y_pred_opt):.4f}")
# print(f"Test MAE: {mean_absolute_error(y_test, y_pred_opt):.4f}")


# --- 2. Yaklaşım: Random Forest (En Güçlü Çözüm) ---
# Tek bir ağaç yerine 100 ağaçtan oluşan bir topluluk modeli kuruyoruz
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# Random Forest tahminleri
y_pred_rf = rf_model.predict(X_test)

# print("\n--- Random Forest Regressor Sonuçları ---")
# print(f"Eğitim R2 Skoru: {rf_model.score(X_train, y_train):.4f}")
# print(f"Test R2 Skoru: {r2_score(y_test, y_pred_rf):.4f}")
# print(f"Test MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}")


# --- 3. Özellik Önemi Analizi (Hangi Değişken Daha Etkili?) ---
# Modelin hangi değişkenlere güvenerek karar verdiğini görelim
importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)
# print("\n--- En Önemli 5 Değişken (Random Forest) ---")
# print(importances.head(5))


# Denge Kurulması: İlk model "High Variance" (Yüksek Varyans) problemine sahipti. Parametre optimizasyonu ile varyansı düşürüp "Bias"ı (Yanlılık) hafifçe artırarak daha dengeli ve genellenebilir bir model elde ettik.

# Verinin Doğası: Enerji tüketimi gibi değişken verilerde (örneğin biri aniden fırını açtığında), sensör verileri (nem, sıcaklık) bu ani artışı açıklamakta yetersiz kalabilir. lag_1 bu boşluğu doldurarak modelin "eylemsizlik" prensibini öğrenmesini sağladı.

# Hata Payı (MAE): Yaklaşık 30 Watt'lık bir hata payımız var. Veri setindeki ortalama Appliances tüketimine bakarak bu hatanın yüzde kaçlık bir sapmaya denk geldiğini belirlemek, projenin iş değerini (business value) ortaya koyacaktır.


# Daha net bir görünüm için test setinden rastgele 100-200 örnek seçelim
# plt.figure(figsize=(16, 6))
# plt.plot(y_test.values[:150], label='Gerçek Tüketim (Actual)', color='#2c3e50', linewidth=2, alpha=0.8)
# plt.plot(y_pred_rf[:150], label='RF Tahmini (Predicted)', color='#e74c3c', linestyle='--', linewidth=2)

# plt.title("Gerçek vs Tahmin Edilen Enerji Tüketimi (İlk 150 Gözlem)", fontsize=14)
# plt.xlabel("Zaman Dilimi (Her 10 Dakika)", fontsize=12)
# plt.ylabel("Watt-saat (Wh)", fontsize=12)
# plt.legend(loc='upper right')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# Özellik önemlerini görselleştirelim
importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=True) # Barh için küçükten büyüğe

plt.figure(figsize=(10, 8))
plt.barh(importances['Feature'], importances['Importance'], color='#3498db')
plt.title("Model İçin En Önemli Değişkenler (Feature Importance)", fontsize=14)
plt.xlabel("Önem Skoru", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()