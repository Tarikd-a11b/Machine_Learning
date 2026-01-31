
# import numpy as np
# import pandas as pd

# # NumPy ile rastgele veri oluşturma
# np.random.seed(42) # Tekrar çalıştırıldığında aynı sonucu almak için
# yaslar = np.random.randint(20, 50, size=10) # 20-50 arası 10 rastgele yaş
# maaslar = np.random.randint(30000, 70000, size=10)
# sehirler = ['Istanbul', 'Ankara', 'Izmir'] * 4 # Fazla şehir üretip kesme
# sehirler = sehirler[:10]

# # Pandas DataFrame oluşturma
# veri = pd.DataFrame({
#     'Yaş': yaslar,
#     'Maaş': maaslar,
#     'Şehir': sehirler
# })

# print("--- DataFrame'in İlk 5 Satırı ---")
# print(veri.head())

# print("\n--- Özet İstatistikler (Sayısal Sütunlar) ---")
# print(veri.describe())

# print("\n--- İstanbul'daki Çalışanları Filtreleme ---")
# istanbul_calisanlari = veri[veri['Şehir'] == 'Istanbul']
# print(istanbul_calisanlari)

# print("\n--- Şehirlere Göre Ortalama Maaş ---")
# ortalama_maas_sehire_gore = veri.groupby('Şehir')['Maaş'].mean()

# print(ortalama_maas_sehire_gore)


######## #Bu örnekte, bir e-ticaret şirketinin 2024 yılının ilk çeyreğindeki (Q1) satış performansını ve müşteri davranışlarını inceleyeceğiz.

import numpy as np
import pandas as pd

# Sabit Değerler
urun_kategorileri = ['Elektronik', 'Giyim', 'Ev & Yaşam', 'Kozmetik']
sehirler = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya']

# Rastgele Veri Üretimi
np.random.seed(42)
satıs_miktari = np.random.randint(50, 5000, size=20) # Sipariş başına TL cinsinden miktar
adet = np.random.randint(1, 10, size=20) # Satın alınan ürün adedi
musteri_yasi = np.random.randint(18, 55, size=20)
kategori = np.random.choice(urun_kategorileri, size=20)
sehir = np.random.choice(sehirler, size=20)
iade = np.random.choice([True, False], p=[0.2, 0.8], size=20) # %20 iade oranı

# DataFrame Oluşturma
df_eticaret = pd.DataFrame({
    'SiparisID': range(101, 121),
    'Kategori': kategori,
    'SatisMiktari_TL': satıs_miktari,
    'UrunAdedi': adet,
    'MusteriYasi': musteri_yasi,
    'Sehir': sehir,
    'Iade': iade
})

# İade olan siparişlerin miktarını 0 yapalım (Gerçek hayatta bu daha karmaşık olabilir)

df_eticaret.loc[df_eticaret['Iade'] == True, 'SatisMiktari_TL'] = 0

print("--- İlk 5 Satır ve Veri Yapısı ---")
print(df_eticaret.head())
print("\n--- Veri Tipleri ---")
print(df_eticaret.info())


import numpy as np
import pandas as pd

# Sabit Değerler
urun_kategorileri = ['Elektronik', 'Giyim', 'Ev & Yaşam', 'Kozmetik']
sehirler = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya']

# Rastgele Veri Üretimi
np.random.seed(42)
satıs_miktari = np.random.randint(50, 5000, size=20) # Sipariş başına TL cinsinden miktar
adet = np.random.randint(1, 10, size=20) # Satın alınan ürün adedi
musteri_yasi = np.random.randint(18, 55, size=20)
kategori = np.random.choice(urun_kategorileri, size=20)
sehir = np.random.choice(sehirler, size=20)
iade = np.random.choice([True, False], p=[0.2, 0.8], size=20) # %20 iade oranı

# DataFrame Oluşturma
df_eticaret = pd.DataFrame({
    'SiparisID': range(101, 121),
    'Kategori': kategori,
    'SatisMiktari_TL': satıs_miktari,
    'UrunAdedi': adet,
    'MusteriYasi': musteri_yasi,
    'Sehir': sehir,
    'Iade': iade
})

# İade olan siparişlerin miktarını 0 yapalım (Gerçek hayatta bu daha karmaşık olabilir)
df_eticaret.loc[df_eticaret['Iade'] == True, 'SatisMiktari_TL'] = 0

print("--- İlk 5 Satır ve Veri Yapısı ---")
print(df_eticaret.head())
print("\n--- Veri Tipleri ---")
print(df_eticaret.info())



print(df_eticaret[['SatisMiktari_TL', 'UrunAdedi', 'MusteriYasi']].describe())


# Kategoriye göre toplam satışı bulma
kategori_satis = df_eticaret.groupby('Kategori')['SatisMiktari_TL'].sum().sort_values(ascending=False)
print("\n--- Kategoriye Göre Toplam Satış Miktarı (TL) ---")
print(kategori_satis)

# Şehirlere göre ortalama ürün adedi bulma
sehir_urun_ort = df_eticaret.groupby('Sehir')['UrunAdedi'].mean().sort_values(ascending=False)
print("\n--- Şehirlere Göre Ortalama Ürün Adedi ---")
print(sehir_urun_ort)



# Sadece 2000 TL üzeri satışları filtreleyelim
yuksek_satislar = df_eticaret[df_eticaret['SatisMiktari_TL'] > 2000]

# Bu yüksek satışların yapıldığı müşterilerin ortalama yaşını bulalım
ortalama_yas_yuksek_satis = yuksek_satislar['MusteriYasi'].mean()

print(f"\n--- Yüksek Satış (2000 TL Üzeri) Yapan Müşterilerin Ortalama Yaşı: {ortalama_yas_yuksek_satis:.2f} ---")

import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn'un varsayılan stilini ayarlayalım
sns.set_style("whitegrid")

# Veri setimiz: df_eticaret (önceki adımdan)

# Önceki adımdan kategoriye göre toplam satış miktarını hesaplayalım
kategori_satis = df_eticaret.groupby('Kategori')['SatisMiktari_TL'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(10, 6))
# Seaborn'un barplot fonksiyonunu kullanıyoruz
sns.barplot(x='Kategori', y='SatisMiktari_TL', data=kategori_satis, palette="viridis")

plt.title('Kategoriye Göre Toplam Satış Miktarı (TL)', fontsize=16)
plt.xlabel('Ürün Kategorisi')
plt.ylabel('Toplam Satış Miktarı (TL)')
plt.xticks(rotation=45) # Kategori isimlerinin üst üste binmesini engellemek için
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
# Seaborn'un histplot fonksiyonu, verinin dağılımını gösterir
sns.histplot(df_eticaret['MusteriYasi'], bins=10, kde=True, color='teal')

plt.title('Müşteri Yaş Dağılımı', fontsize=16)
plt.xlabel('Müşteri Yaşı')
plt.ylabel('Müşteri Sayısı (Frekans)')
plt.show()


plt.figure(figsize=(8, 5))
# Saçılım grafiği, iki değişken arasındaki ilişkiyi gösterir
sns.scatterplot(x='MusteriYasi', y='SatisMiktari_TL', data=df_eticaret, hue='Kategori', palette='Set1', s=100)

plt.title('Yaş ve Satış Miktarı İlişkisi', fontsize=16)
plt.xlabel('Müşteri Yaşı')
plt.ylabel('Satış Miktarı (TL)')
plt.legend(title='Kategori', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()