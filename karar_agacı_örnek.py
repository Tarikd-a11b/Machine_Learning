import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# 1. VERİ SETİNİ OLUŞTURMA (Slayt Sayfa 17'deki Tablo)
# Slayttaki veriyi birebir buraya aktarıyoruz.
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# 2. VERİYİ SAYISALLAŞTIRMA (ENCODING)
# Makine "Sunny" nedir bilmez, ona 0, 1, 2 demeliyiz.
# Mapping (Haritalama) yapıyoruz:
# Outlook: Sunny(0), Overcast(1), Rain(2)
# Humidity: High(0), Normal(1)
# Wind: Weak(0), Strong(1)

df['Outlook_Encoded'] = df['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
df['Humidity_Encoded'] = df['Humidity'].map({'High': 0, 'Normal': 1})
df['Wind_Encoded'] = df['Wind'].map({'Weak': 0, 'Strong': 1})
df['Play_Encoded'] = df['PlayTennis'].map({'No': 0, 'Yes': 1})

# Girdi (X) ve Çıktı (y) Ayrımı
X = df[['Outlook_Encoded', 'Humidity_Encoded', 'Wind_Encoded']]
y = df['Play_Encoded']

# 3. MODELİ KURMA
# 'entropy' kriterini kullanıyoruz (Slaytlardaki Bilgi Kazancı/Information Gain yöntemi)
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y)

# 4. AĞACI GÖRSELLEŞTİRME VE ÇIKTI ALMA
print("--- Makinenin Oluşturduğu Kurallar ---\n")
print(export_text(clf, feature_names=['Outlook', 'Humidity', 'Wind']))

# Grafik olarak da çizelim
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=['Outlook', 'Humidity', 'Wind'], class_names=['No', 'Yes'], filled=True)
plt.show()