
import pandas as pd
import numpy as np

# 1. Orijinal veriyi yükleyin
# Dosya adınız farklıysa burayı güncelleyin
df = pd.read_csv(r"C:\Users\bilal\Desktop\Datasets\Student_Performance.xlsx")

# 2. Rastgeleliği sabitleyin (her seferinde aynı sonucu almak için)
np.random.seed(42)

# 3. Dengesizleştirme Olasılıklarını Tanımlayın
def get_gender_keep_prob(gender):
    if gender == 'female': return 1.0       # Kadınlar aynen kalsın
    elif gender == 'male': return 0.9       # Erkekleri %10 azalt
    elif gender == 'other': return 0.05     # Diğer cinsiyeti %95 azalt (Azınlık sınıfı yap)
    return 1.0

def get_school_keep_prob(school):
    if school == 'public': return 1.0       # Devlet okulu aynen kalsın
    elif school == 'private': return 0.4    # Özel okulu %60 azalt
    return 1.0

# 4. Filtreleme İşlemi
df['gender_rand'] = np.random.rand(len(df))
df['school_rand'] = np.random.rand(len(df))

condition_gender = df.apply(lambda row: row['gender_rand'] < get_gender_keep_prob(row['gender']), axis=1)
condition_school = df.apply(lambda row: row['school_rand'] < get_school_keep_prob(row['school_type']), axis=1)

df_imbalanced = df[condition_gender & condition_school].copy()
df_imbalanced = df_imbalanced.drop(columns=['gender_rand', 'school_rand'])

# 5. Yeni dosyayı kaydedin
df_imbalanced.to_csv('Student_Performance_Dengesiz.csv', index=False)
print("Dosya oluşturuldu: Student_Performance_Dengesiz.csv")