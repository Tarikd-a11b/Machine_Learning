import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =================================================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# =================================================================
df = pd.read_csv('16-diabetes.csv')

# İstediğin üzerine 'Insulin' ve 'SkinThickness' sütunlarını tamamen çıkarıyoruz.
# Bu sütunlar yüksek oranda eksik (0) veri içerdiği için modelin gürültüden arınmasını sağlar.
X = df.drop(["Outcome", "Insulin", "SkinThickness"], axis=1)
y = df["Outcome"]

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15, test_size=0.20)

# =================================================================
# 2. VERİ TEMİZLEME (IMPUTATION)
# =================================================================
# Geri kalan kritik sütunlardaki (Glucose, BloodPressure, BMI) 0 değerlerini 
# eğitim setinin medyanı ile dolduruyoruz.
columns_to_fill = ["Glucose", "BloodPressure", "BMI"]

for col in columns_to_fill:
    # 0 olmayan değerlerin medyanını al
    median_value = X_train[X_train[col] != 0][col].median()
    X_train[col] = X_train[col].replace(0, median_value)
    X_test[col] = X_test[col].replace(0, median_value)

print(f"--- Veri Hazırlığı Tamamlandı ---")
print(f"Kullanılan Özellikler: {list(X_train.columns)}")
print(f"Eğitim Seti Boyutu: {X_train.shape}\n")

# =================================================================
# 3. MODEL KARŞILAŞTIRMASI
# =================================================================
# Naive Bayes ve Logistic Regression dahil tüm modelleri bir listede topluyoruz.
models = [
    ('Naive Bayes (Gaussian)', GaussianNB()),
    ('Logistic Regression', LogisticRegression(max_iter=1000, solver='lbfgs')),
    ('AdaBoost (Optimized)', AdaBoostClassifier(learning_rate=0.5, n_estimators=120, random_state=15)),
    ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=15)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('SVC', SVC(probability=True, random_state=15))
]

results = []

print("--- Modellerin Performans Analizi ---")
for name, model in models:
    # Modeli eğit
    model.fit(X_train, y_train)
    # Tahmin yap
    y_pred = model.predict(X_test)
    
    # Accuracy skorunu hesapla
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})
    
    print(f"{name:25} | Accuracy: {acc:.4f}")

# =================================================================
# 4. SONUÇLARIN ÖZETİ VE DETAYLI RAPORLAR
# =================================================================
# Sonuçları DataFrame'e döküp sıralayalım
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print("\n--- Başarı Sıralaması ---")
print(results_df)

print("\n" + "="*30 + " KRİTİK MODEL RAPORLARI " + "="*30)

# 1. Naive Bayes Raporu
print("\n[ NAIVE BAYES DETAYLI ANALİZ ]")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print(classification_report(y_test, nb_pred))

# 2. Logistic Regression Raporu
print("\n[ LOGISTIC REGRESSION DETAYLI ANALİZ ]")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print(classification_report(y_test, lr_pred))

print("="*84)