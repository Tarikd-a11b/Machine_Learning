
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. Sabit Tanımlar (Önceki SQL Tablolarımızdaki gibi) ---
urunler = {
    101: 'Akıllı Termostat X1',
    102: 'Robot Süpürge Y2',
    103: 'Kablosuz Kulaklık Z3',
    104: 'Güneş Paneli Sistemi A4' # Yeni ürün ekleyelim
}

departmanlar = {
    1: 'Ar-Ge ve Tasarım',
    2: 'Hammadde Tedariği',
    3: 'Montaj ve Üretim Bandı',
    4: 'Kalite Kontrol ve Test',
    5: 'Paketleme ve Sevkiyat'
}

harcama_tipleri = [
    'İşçilik', 'Malzeme', 'Enerji', 'Makine Bakım', 'Yazılım Lisans', 
    'Prototip Geliştirme', 'Test Ekipmanı', 'Lojistik'
]

# --- 2. Rastgele Veri Üretme Parametreleri ---
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 1, 30)
NUM_RECORDS = 500 # Toplam satır sayısı

# Üretim Adedi dağılımı (Farklı ürünler için farklı adetler)
adet_dagilimi = {
    101: 500, # Termostat
    102: 300, # Süpürge
    103: 1500, # Kulaklık (Daha çok üretilir)
    104: 100 # Güneş Paneli (Daha az üretilir)
}

# Tutar (Maliyet) dağılımı: Daha gerçekçi bir simülasyon için
# Ar-Ge ve Hammadde, genellikle diğerlerinden daha pahalıdır
tutar_dagilimi = {
    1: (10000, 50000), # Ar-Ge
    2: (20000, 75000), # Hammadde
    3: (5000, 30000), # Montaj
    4: (3000, 15000), # Kalite Kontrol
    5: (4000, 20000)  # Paketleme
}

# --- 3. Veri Setini Oluşturma ---
data = []
for i in range(1, NUM_RECORDS + 1):
    urun_id = np.random.choice(list(urunler.keys()))
    departman_id = np.random.choice(list(departmanlar.keys()))
    harcama_tipi = np.random.choice(harcama_tipleri)
    
    # Tarih aralığında rastgele bir tarih seçme
    random_days = np.random.randint(0, (END_DATE - START_DATE).days)
    tarih = START_DATE + timedelta(days=random_days)
    
    # Departman bazında rastgele tutar oluşturma
    min_tutar, max_tutar = tutar_dagilimi[departman_id]
    tutar = round(np.random.uniform(min_tutar, max_tutar), 2)
    
    # Ürün bazında üretilen adet bilgisini alma
    adet = adet_dagilimi[urun_id]
    
    data.append([i, urun_id, departman_id, harcama_tipi, tutar, tarih.strftime('%Y-%m-%d'), adet])

# DataFrame oluşturma
columns = ['HarcamaID', 'UrunID', 'DepartmanID', 'HarcamaTipi', 'Tutar', 'Tarih', 'Adet']
df_harcamalar = pd.DataFrame(data, columns=columns)


# --- 4. Yardımcı Tabloları Oluşturma ---

# Urunler DataFrame
df_urunler = pd.DataFrame(
    list(urunler.items()), 
    columns=['UrunID', 'UrunAdi']
)
df_urunler['Kategori'] = np.select(
    [df_urunler['UrunID'].isin([101, 103]), df_urunler['UrunID'].isin([102]), df_urunler['UrunID'].isin([104])],
    ['Elektronik', 'Ev Aletleri', 'Yenilenebilir Enerji'],
    default='Diğer'
)

# Departmanlar DataFrame
df_departmanlar = pd.DataFrame(
    list(departmanlar.items()), 
    columns=['DepartmanID', 'DepartmanAdi']
)


# --- 5. CSV Dosyalarına Kaydetme ---

df_harcamalar.to_csv('UretimHarcamalari.csv', index=False)
df_urunler.to_csv('Urunler.csv', index=False)
df_departmanlar.to_csv('Departmanlar.csv', index=False)

print(f"Başarıyla {NUM_RECORDS} satırlık 'UretimHarcamalari.csv' dosyası oluşturuldu.")
print("Yardımcı tablolar (Urunler.csv, Departmanlar.csv) da oluşturuldu.")
print("Veri setini SQL veya Power BI'a aktarmaya hazırsınız.")

# Verinin ilk 5 satırını görme
print("\nÖrnek Veri:")
print(df_harcamalar.head())