
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\bilal\Desktop\Titanic-Dataset.csv")
print(df.head())
print(df)
print(df.info())
# Handling missing values

print(df.isnull().sum())

print(df.dropna(axis=1))
#eksik columnları sildik ama satırlar bundan etkilenmemesiçin axis=1 dedik age,embarked ve cabin columnlarında eksik veriler vardı.
#ama bu durum veri kaybına yol açabilir.özellikle age columnunu hariç diğer colunmların çok eksik verisi yoktu.Bundan ötürü veri doldruma daha mantıklı olabilir.

df["age_mean"] = df['Age'].fillna(df['Age'].mean())
print(df[['Age', 'age_mean']])

df.boxplot(column='Age')
plt.show()


mode_valeue = df[df["Embarked"].notnull()]["Embarked"].mode()[0]
print(mode_valeue)

df["Embarked_filled"] = df["Embarked"].fillna(mode_valeue)

print(df[["Embarked", "Embarked_filled"]])