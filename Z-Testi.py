import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Sonuçlar her seferinde aynı çıksın diye sabitleyelim
np.random.seed(42)

# 1. ADIM: Veri Oluşturma (Simülasyon)
# Beşiktaş: Ortalaması 1000 TL, Standart Sapması 100 TL olan 50 günlük ciro
besiktas_satis = np.random.normal(1000, 100, 50)

# Kadıköy: Ortalaması 1060 TL, Standart Sapması 110 TL olan 50 günlük ciro
# (Kadıköy'ü bilerek biraz daha yüksek yaptık, bakalım test yakalayacak mı?)
kadikoy_satis = np.random.normal(1060, 110, 50)

# 2. ADIM: T-Testi Uygulama
# scipy kütüphanesi bizim için karmaşık matematiği tek satırda yapar
t_stat, p_value = stats.ttest_ind(kadikoy_satis, besiktas_satis)

# 3. ADIM: Sonuçları Yazdırma
print(f"Beşiktaş Ortalama Satış: {besiktas_satis.mean():.2f} TL")
print(f"Kadıköy Ortalama Satış: {kadikoy_satis.mean():.2f} TL")
print(f"Fark: {kadikoy_satis.mean() - besiktas_satis.mean():.2f} TL")
print("-" * 30)
print(f"T-İstatistiği (Sinyal Gücü): {t_stat:.4f}")
print(f"P-Değeri (Şans İhtimali): {p_value:.6f}")
print("-" * 30)

# Karar Mekanizması
alpha = 0.05 # %5 yanılma payı sınırımız
if p_value < alpha:
    print("SONUÇ: H0 Reddedildi! Fark İstatistiksel Olarak ANLAMLI.")
    print("Yorum: Kadıköy gerçekten daha iyi satıyor, prim verilebilir.")
else:
    print("SONUÇ: H0 Reddedilemedi. Fark Tesadüfi Olabilir.")
    print("Yorum: Aradaki fark şans eseri oluşmuş olabilir, hemen karar verme.")

# 4. ADIM: Görselleştirme
plt.figure(figsize=(10, 6))
sns.kdeplot(besiktas_satis, shade=True, label="Beşiktaş", color="blue")
sns.kdeplot(kadikoy_satis, shade=True, label="Kadıköy", color="red")
plt.axvline(besiktas_satis.mean(), color='blue', linestyle='--')
plt.axvline(kadikoy_satis.mean(), color='red', linestyle='--')
plt.title("İki Şubenin Satış Dağılımı Karşılaştırması")
plt.legend()
plt.show()