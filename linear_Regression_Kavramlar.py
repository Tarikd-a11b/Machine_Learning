
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# 1. Veri Oluşturma (Reklam Harcaması ve Satış)
np.random.seed(42)
reklam = np.random.normal(100, 20, 100) # Bağımsız Değişken (X)
satis = 5 + 1.5 * reklam + np.random.normal(0, 10, 100) # Bağımlı Değişken (Y)

df = pd.DataFrame({'Reklam': reklam, 'Satis': satis})

# 2. Model Kurma
X = sm.add_constant(df['Reklam']) # Sabit terim (intercept) ekleme
model = sm.OLS(df['Satis'], X).fit()

# 3. SST, SSE, SSR Hesaplama (Görselindeki formüllere göre)
y_gercek = df['Satis']
y_tahmin = model.predict(X)
y_ortalama = np.mean(y_gercek)

SST = np.sum((y_gercek - y_ortalama)**2)
SSE = np.sum((y_tahmin - y_ortalama)**2) # Açıklanan (Görselindeki tanım)
SSR = np.sum((y_gercek - y_tahmin)**2) # Hata (Görselindeki tanım)

print(f"Toplam Değişkenlik (SST): {SST:.2f}")
print(f"Açıklanan Değişkenlik (SSE): {SSE:.2f}")
print(f"Hata Değişkenliği (SSR): {SSR:.2f}")
print(f"R-Kare (SSE/SST): {SSE/SST:.4f}")

# 4. Varsayım Kontrolleri (Görselleştirme)
residuals = model.resid # Hatalar

plt.figure(figsize=(12, 5))

# Homoscedasticity Kontrolü
plt.subplot(1, 2, 1)
plt.scatter(y_tahmin, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Homoscedasticity Kontrolü\n(Hatalar vs Tahminler)')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Hatalar (Residuals)')

# Normality Kontrolü (Q-Q Plot)
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normality Kontrolü\n(Q-Q Plot)')

plt.tight_layout()
plt.show()