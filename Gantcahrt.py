import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime

# Verileri görselden manuel olarak çıkarıp listeye döküyoruz
data = [
    # Planlama
    {"Task": "Proje Kapsamının Hazırlanması", "Start": "29.09.2025", "End": "01.10.2025", "Phase": "Planlama"},
    {"Task": "Hedef Kitle ve Pazar Analizi", "Start": "01.10.2025", "End": "05.10.2025", "Phase": "Planlama"},
    {"Task": "PHP ve MySQL Ortam Kurulum Planı", "Start": "05.10.2025", "End": "08.10.2025", "Phase": "Planlama"},
    # Analiz
    {"Task": "SME ve Müşteri Gereksinimleri", "Start": "08.10.2025", "End": "12.10.2025", "Phase": "Analiz"},
    {"Task": "Veritabanı Normalizasyon Analizi", "Start": "12.10.2025", "End": "15.10.2025", "Phase": "Analiz"},
    {"Task": "Bildirim Altyapı Analizi", "Start": "15.10.2025", "End": "18.10.2025", "Phase": "Analiz"},
    # Tasarım
    {"Task": "MySQL Veritabanı Şeması", "Start": "20.10.2025", "End": "24.10.2025", "Phase": "Tasarım"},
    {"Task": "SME Paneli Web Arayüz Tasarımı", "Start": "24.10.2025", "End": "28.11.2025", "Phase": "Tasarım"}, # Görseldeki tarih kullanıldı
    {"Task": "Müşteri Randevu Sayfası Tasarımı", "Start": "28.11.2025", "End": "02.12.2025", "Phase": "Tasarım"},
    # Geliştirme
    {"Task": "MySQL Veritabanı Kurulumu", "Start": "02.12.2025", "End": "06.12.2025", "Phase": "Geliştirme"},
    {"Task": "SME Yönetim Paneli Backend", "Start": "06.12.2025", "End": "14.12.2025", "Phase": "Geliştirme"},
    {"Task": "Randevu Alma Motoru Kodlaması", "Start": "14.12.2025", "End": "22.12.2025", "Phase": "Geliştirme"},
    {"Task": "E-posta Bildirim Servisi Entegrasyonu", "Start": "22.12.2025", "End": "26.12.2025", "Phase": "Geliştirme"},
    {"Task": "Frontend Entegrasyonu", "Start": "26.12.2025", "End": "30.12.2025", "Phase": "Geliştirme"},
    # Test ve Kapanış
    {"Task": "Fonksiyonel Testler", "Start": "30.12.2025", "End": "02.01.2026", "Phase": "Test ve Kapanış"},
    {"Task": "Hata Düzeltmeleri (Bug Fixing)", "Start": "02.01.2026", "End": "05.01.2026", "Phase": "Test ve Kapanış"},
    {"Task": "Proje Raporu ve Teslim", "Start": "05.01.2026", "End": "07.01.2026", "Phase": "Test ve Kapanış"}
]

# DataFrame oluşturma
df = pd.DataFrame(data)

# Tarih formatını datetime objesine çevirme
df['Start'] = pd.to_datetime(df['Start'], format='%d.%m.%Y')
df['End'] = pd.to_datetime(df['End'], format='%d.%m.%Y')

# Süreyi hesaplama
df['Duration'] = df['End'] - df['Start']

# Renk paleti
colors = {
    "Planlama": "#3498db",  # Mavi
    "Analiz": "#f1c40f",    # Sarı
    "Tasarım": "#e67e22",   # Turuncu
    "Geliştirme": "#2ecc71",# Yeşil
    "Test ve Kapanış": "#e74c3c" # Kırmızı
}
c_dict = df['Phase'].map(colors)

# Grafik oluşturma
fig, ax = plt.subplots(figsize=(12, 10))

# Barların çizimi
# Y eksenini ters çeviriyoruz ki ilk görev en üstte olsun
y_pos = range(len(df))
ax.barh(y_pos, df['Duration'].dt.days, left=df['Start'], color=c_dict, height=0.6, align='center')

# Eksen formatlaması
ax.set_yticks(y_pos)
ax.set_yticklabels(df['Task'])
ax.invert_yaxis()  # Listeyi yukarıdan aşağıya sırala

# X ekseni tarih formatı
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7)) # Her 7 günde bir işaret
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))

# Başlık ve Etiketler
plt.title('Proje Yönetim Süreci (Gantt Şeması)', fontsize=14, pad=20)
plt.xlabel('Tarih', fontsize=12)
plt.grid(True, axis='x', linestyle='--', alpha=0.7)

# Legend (Renk Açıklaması) oluşturma
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in colors]
plt.legend(handles, colors.keys(), loc='lower right')

plt.tight_layout()
plt.show()
# Grafiği kaydetme
fig.savefig('Gantt_Chart_Proje_Yonetimi.png', dpi=300)
plt.close(fig)


