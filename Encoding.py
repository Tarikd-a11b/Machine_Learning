
#one hot encoding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder


df = pd.read_csv(r"C:\Users\bilal\Desktop\Titanic-Dataset.csv")
print(df.head())

print(df[["Sex","Pclass"]].isna().sum())

print(df["Sex"].value_counts())
print(df["Pclass"].value_counts())

df_onehot=pd.get_dummies(df,columns=["Sex","Pclass"],drop_first=True)
print(df_onehot.head())

print(df_onehot.columns)

#label encoding
label_encoder = LabelEncoder()
df_label = df.copy()
df_label["Sex"] = label_encoder.fit_transform(df_label["Sex"])
print(df_label.head())

#ordinal encoding

#df_ordinal = df.copy()
#class_order = ["Third", "Second", "First"]  # Define the order for 'Pclass'
#ordinal_encoder = OrdinalEncoder(categories=[class_order])
# df_ordinal[["Pclass"]] = ordinal_encoder.fit_transform(df_ordinal[["Pclass"]])
# print(df_ordinal.head())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df["Sex"].value_counts().plot(kind="bar", ax=axes[0], title="Original Categorical")
df_label["Sex"].value_counts().plot(kind="bar", ax=axes[1], title="Label Encoded")
df_onehot[["Sex_male"]].sum().plot(kind="bar", ax=axes[2], title="One-Hot Encoded")
plt.show()