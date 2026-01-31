
import numpy as np
import matplotlib.pyplot as plt

# 1. ADIM: Kaotik (Düzgün) Bir Popülasyon Oluşturma
# 1 ile 6 arasında rastgele 100.000 adet sayı (Zar atma simülasyonu)
# Bu dağılım 'Normal' değildir, dümdüzdür (Uniform).
populasyon = np.random.randint(1, 7, 100000)

# 2. ADIM: Örneklem Ortalamalarını Toplama
orneklem_ortalamalari = []
simulasyon_sayisi = 1000   # Bu işlemi 1000 kere tekrarla
n = 30                     # Her seferinde 30 tane zar seç (n=30 kuralı)

for i in range(simulasyon_sayisi):
    # Popülasyondan rastgele n tane sayı seç
    orneklem = np.random.choice(populasyon, n)
    # Bu grubun ortalamasını al ve listeye ekle
    orneklem_ortalamalari.append(np.mean(orneklem))

# 3. ADIM: Görselleştirme
plt.figure(figsize=(10, 6))

# Histogramı çizelim
plt.hist(orneklem_ortalamalari, bins=30, density=True, alpha=0.7, color='#3498db', edgecolor='black')

# Teorik ortalamayı (Zar için 3.5) kırmızı çizgiyle gösterelim
plt.axvline(3.5, color='red', linestyle='dashed', linewidth=2, label='Gerçek Ortalama (3.5)')

plt.title('Merkezi Limit Teoremi İspatı\n(Zar Atma Örneği)', fontsize=14)
plt.xlabel('Örneklem Ortalamaları', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.5)

plt.show()