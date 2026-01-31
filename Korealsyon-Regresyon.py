import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Veri Sabitleme
np.random.seed(42)

# 1. ADIM: Veri Oluşturma
# 100 günlük reklam harcaması (X ekseni): 50 TL ile 200 TL arası
reklam_harcamasi = np.random.randint(50, 200, 100)

# Satışlar (Y ekseni): Reklam arttıkça artacak şekilde formülize edelim
# Formül: (Reklam x 2.5) + (Rastgelelik/Gürültü) + (Sabit Müşteri)
# Yani her 1 TL reklam, satışı yaklaşık 2.5 TL artırıyor.
satislar = (reklam_harcamasi * 2.5) + np.random.normal(0, 30, 100) + 100

# 2. ADIM: Korelasyon Hesaplama (Pearson Katsayısı)
korelasyon, p_value = stats.pearsonr(reklam_harcamasi, satislar)

# 3. ADIM: Regresyon Çizgisi ve Görselleştirme
plt.figure(figsize=(10, 6))

# Regresyon grafiği (Scatter Plot + Çizgi)
sns.regplot(x=reklam_harcamasi, y=satislar, ci=None, 
            scatter_kws={'color':'blue', 'alpha':0.6}, 
            line_kws={'color':'red', 'linewidth':3})

plt.title(f'Reklam ve Satış İlişkisi\nKorelasyon (r): {korelasyon:.2f}', fontsize=15)
plt.xlabel('Reklam Harcaması (TL)', fontsize=12)
plt.ylabel('Satış Adedi/Tutarı (TL)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# İlişkinin Gücünü Yorumlayalım
print(f"Korelasyon Katsayısı (r): {korelasyon:.4f}")