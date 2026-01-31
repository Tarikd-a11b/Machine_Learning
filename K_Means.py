
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Veri Oluşturma (Senaryo: Harcama vs. Alışveriş Sıklığı)
np.random.seed(42)
n_musteri = 300
harcama = np.random.normal(5000, 1500, n_musteri)
sıklık = np.random.normal(25, 8, n_musteri)

df = pd.DataFrame({'Harcama': harcama, 'Sıklık': sıklık})
df = df[(df['Harcama'] > 0) & (df['Sıklık'] > 0)] # Mantıklı değerler kalsın

# 2. ÖNEMLİ: Ölçeklendirme (Scaling)
# Harcama binlerce TL iken sıklık 10-20 arasındadır. 
# K-Means uzaklık baktığı için büyük sayılara (TL) öncelik vermesin diye ikisini de 0-1 arasına yaklaştırıyoruz.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 3. Dirsek Yöntemi (Elbow Method) ile İdeal K Sayısını Bulma
wcss = [] # Within-Cluster Sum of Squares (Küme içi hata kareler toplamı)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Grafiği çizdirelim
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='purple')
plt.title('İdeal Küme Sayısı (Dirsek Yöntemi)')
plt.xlabel('Küme Sayısı (K)')
plt.ylabel('Hata Değeri (WCSS)')
plt.grid(True)
plt.show()

# 4. K-Means Algoritmasını Uygulama (K=3 seçtik)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Kume'] = kmeans.fit_predict(scaled_data)

# 5. Görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Harcama', y='Sıklık', hue='Kume', palette='bright', s=100)

# Merkez noktalarını da işaretleyelim (Ölçeklendirilmiş hali geri çevirerek)
merkezler = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(merkezler[:, 0], merkezler[:, 1], s=300, c='black', marker='X', label='Küme Merkezleri')

plt.title('K-Means Müşteri Segmentasyonu Sonucu')
plt.xlabel('Yıllık Harcama (TL)')
plt.ylabel('Alışveriş Sıklığı (Adet)')
plt.legend()
plt.show()

# Grupların ortalamasına bakalım (Yorumlama için)
print("\n--- Küme Ortalamaları ---")
print(df.groupby('Kume').mean())