import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

df = pd.read_csv("Exam_Score_Prediction.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# Data Visualization
plt.figure(figsize=(10, 6))

# scatter_kws: Noktaların özelliklerini (şeffaflık gibi) ayarlar
# line_kws: Çizginin özelliklerini ayarlar
sns.regplot(x='study_hours', y='exam_score', data=df, 
            scatter_kws={'alpha': 0.1, 's': 10}, 
            line_kws={'color': 'red'})

plt.title('Çalışma Saati vs Sınav Puanı (Eğilim Çizgisi ile)')
plt.xlabel('Çalışma Saati')
plt.ylabel('Sınav Puanı')
plt.show()

# Cinsiyet gibi verileri 0 ve 1'e çeviriyoruz
df_encoded = pd.get_dummies(df, drop_first=True)
correlation_matrix = df_encoded.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Matrisi')
plt.show()

# Veriyi Özellikler ve Hedef olarak ayırma
X = df_encoded.drop('exam_score', axis=1)
y = df_encoded['exam_score']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Modelin katsayıları
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# ============= YENİ VERİ İLE TAHMİN YAPMA =============
print("\n" + "="*50)
print("YENİ ÖĞRENCİ VERİSİ İLE SINAV PUANI TAHMİNİ")
print("="*50)

# Veri setindeki sütunları görelim
print("\nVeri setindeki özellikler:")
print(df.columns.tolist())

# Örnek yeni öğrenci verisi
yeni_ogrenci = {
    'Student_ID': 101,  # Örnek: Yeni öğrenci ID'si
    'age': 20,          # Örnek: 20 yaşında
    'study_hours': 5.5,
    'gender': 'Male',  # Örnek: Erkek
    'class_attendance': 90,
    'sleep_hours': 7
        # Örnek: 5.5 saat çalışma
    # Diğer özellikleri buraya ekleyin (örn: 'attendance': 85, 'gender': 'Male', vb.)
}

# Yeni veriyi DataFrame'e çevirme
yeni_veri_df = pd.DataFrame([yeni_ogrenci])

# Aynı encoding işlemini uygulama
yeni_veri_encoded = pd.get_dummies(yeni_veri_df, drop_first=True)

# Eğitim verisindeki tüm sütunları ekleme (eksik olanlar 0 olarak)
for col in X.columns:
    if col not in yeni_veri_encoded.columns:
        yeni_veri_encoded[col] = 0


# Sütunları aynı sıraya getirme
yeni_veri_encoded = yeni_veri_encoded[X.columns]

# Tahmin yapma
tahmin = model.predict(yeni_veri_encoded)

print(f"\nGirilen Öğrenci Bilgileri:")
for key, value in yeni_ogrenci.items():
    print(f"  {key}: {value}")

print(f"\nTahmin Edilen Sınav Puanı: {tahmin[0]:.2f}")
print("="*50)


# Hata Paylarını (Residuals) Hesaplama
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color='r', linestyle='--') # 0 noktasına çizgi çek
plt.title('Hata Analizi (Residual Plot)')
plt.xlabel('Gerçek Puanlar')
plt.ylabel('Hata Miktarı (Gerçek - Tahmin)')
plt.show()


# Başka Bir Model Deniyoruz: Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Modeli Değiştiriyoruz
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin
y_pred_rf = rf_model.predict(X_test)

# Yeni Skorlar
print("\n" + "="*30)
print("RANDOM FOREST SONUÇLARI")
print("="*30)
print(f"R^2 Score: {sk.metrics.r2_score(y_test, y_pred_rf):.4f}")
print(f"MSE: {sk.metrics.mean_squared_error(y_test, y_pred_rf):.2f}")

# Özellik Önem Düzeyleri
feature_importances = pd.DataFrame(rf_model.feature_importances_, X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
print("\nÖzellik Önem Düzeyleri (Random Forest):")
print(feature_importances)
# Özellik Önem Düzeylerini Görselleştirme
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.index, y='Importance', data=feature_importances)
plt.title('Özellik Önem Düzeyleri (Random Forest)')
plt.xlabel('Özellikler')
plt.ylabel('Önem Düzeyi')
plt.xticks(rotation=45)
plt.show()

# Model Karşılaştırması
print("\n" + "="*30)
print("MODEL KARŞILAŞTIRMASI")
print("="*30)
models = ['Linear Regression', 'Random Forest']
r2_scores = [r2, sk.metrics.r2_score(y_test, y_pred_rf)]
comparison_df = pd.DataFrame({'Model': models, 'R^2 Score': r2_scores})
print(comparison_df)
plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='R^2 Score', data=comparison_df)
plt.title('Model Karşılaştırması (R^2 Score)')
plt.ylim(0, 1)
plt.show()

# Kullanıcıdan veri alma (Kodun en sonuna eklenebilir)
print("\n--- Kendi Puanını Tahmin Et ---")
try:
    saat = float(input("Günde kaç saat çalışıyorsunuz? (Örn: 5.5): "))
    
    # Mevcut kodunuzdaki yapıyı kullanarak tahmin etme
    yeni_veri_encoded['study_hours'] = saat
    tahmin_puan = rf_model.predict(yeni_veri_encoded)[0] # Random Forest ile tahmin
    
    print(f"\n{saat} saat çalışma ile tahmini puanınız: {tahmin_puan:.2f}")
    if tahmin_puan > 85:
        print("Harika gidiyorsunuz! 🌟")
    elif tahmin_puan < 50:
        print("Biraz daha gayret etmelisiniz. 💪")
        
except ValueError:
    print("Lütfen geçerli bir sayı giriniz.")

import joblib
# En başarılı modeli (Random Forest) kaydedelim
joblib.dump(rf_model, 'basarili_ogrenci_modeli.pkl')
print("Model başarıyla kaydedildi.")