
import numpy as np
import statsmodels.api as sm

# 1. Veri Girişi
# X: Store size (Square Feet)
X = np.array([1726, 1542, 2816, 5555, 1292, 2208, 1313])

# Y: Annual Sales ($1000)
Y = np.array([3681, 3395, 6653, 9543, 3318, 5563, 3760])

# 2. X'e bir sabit (kesme noktası) sütunu ekleme
# Statsmodels kütüphanesi otomatik olarak b0 (kesme noktası) terimini eklemez,
# bu yüzden bunu manuel olarak yapmalıyız.
X = sm.add_constant(X)

# 3. OLS (En Küçük Kareler) Modelini Kurma ve Eğitme
model = sm.OLS(Y, X)
results = model.fit()

# 4. Sonuçları Çıkarma
b0 = results.params[0]  # Kesme Noktası (Intercept)
b1 = results.params[1]  # Eğim (X'in katsayısı)

# 5. Regresyon Denklemini Yazdırma
print("--- Basit Doğrusal Regresyon Sonuçları ---")
print(f"Kesme Noktası (b0): {b0:.4f}")
print(f"Eğim (b1): {b1:.4f}")
print("\nRegresyon Denklemi:")
print(f"Ŷ = {b0:.4f} + {b1:.4f} X")
print("-" * 45)

# İsteğe bağlı olarak, tam istatistiksel özet
# print(results.summary())