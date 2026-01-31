
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv(r"C:\Users\bilal\Downloads\17-googleplaystore.csv")

print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())


print(df['Reviews'].str.isnumeric().sum())  #10841 satırdan oluuşan Review sütununda yalnızca bir satırdaki veri numerik değil.Aşağıdaki kodla bu satırı bulabiliriz.
print(df[~df['Reviews'].str.isnumeric()])  #10841.satırdaki veriyi bulduk.

df_clean=df.copy()  #Veri setinin bir kopyasını oluşturduk.
df_clean=df_clean.drop(10472)  #10472.satırı sildik.

df_clean['Reviews']=df_clean['Reviews'].astype(int)  #Reviews sütunundaki verileri integer veri tipine dönüştürdük.
print(df_clean['Reviews'].dtype)  #Veri tipini kontrol ettik.

df_clean['Size']=df_clean['Size'].replace('Varies with device',np.nan)  #Size sütunundaki 'Varies with device' değerlerini NaN ile değiştirdik.
df_clean['Size']=df_clean['Size'].str.replace('M','000')  #Size sütunundaki 'M' harfini '000' ile değiştirdik.
df_clean['Size']=df_clean['Size'].str.replace('k','')   #Size sütunundaki 'k' harfini boş string ile değiştirdik.

print(df_clean['Size'].unique()) 

df_clean['Size']=df_clean['Size'].astype(float)  #Size sütunundaki verileri float veri tipine dönüştürdük.
print(df_clean['Size'].dtype)  #Veri tipini kontrol ettik.


remove_chars=['+',',','$']  #Temizlemek istediğimiz karakterleri bir liste olarak tanımladık.
cols_to_clean=["Installs","Price"]  #Installs ve Price sütunlarındaki verileri temizlemek için bir liste oluşturduk.

for item in remove_chars:
    for col in cols_to_clean:
        df_clean[col]=df_clean[col].str.replace(item,'')  #Her bir karakteri ilgili sütunlardan kaldırdık.
    
print(df_clean["Price"].unique())  #Price sütunundaki benzersiz değerleri kontrol ettik.
print(df_clean["Installs"].unique())  #Installs sütunundaki benzersiz değerleri kontrol ettik.

df_clean['Price']=df_clean['Price'].astype(float)  #Price sütunundaki verileri float veri tipine dönüştürdük.
df_clean['Installs']=df_clean['Installs'].astype(int)  #Installs sütunundaki verileri integer veri tipine dönüştürdük.
df_clean.describe() 


df_clean['Last Updated']=pd.to_datetime(df_clean['Last Updated'])  #Last Updated sütunundaki verileri datetime veri tipine dönüştürdük.
print(df_clean.head())

df_clean["Year"]=df_clean['Last Updated'].dt.year  #Last Updated sütunundan yılı çıkartarak yeni bir Year sütunu oluşturduk.
print(df_clean.head())


#EDA

print(df_clean[df_clean.duplicated('App')].shape)  #Veri setindeki tekrar eden satırları kontrol ettik.

df_clean=df_clean.drop_duplicates(subset=['App'],keep='first')  #Tekrar eden satırları kaldırdık.

print(df_clean.info())  #Veri setinin bilgi özetini görüntüledik.



numerical_cols=[feature for feature in df_clean.columns if df_clean[feature].dtype!='object']  #Sayısal sütunları belirledik.
print("Numerical columns:",numerical_cols)
categorical_cols=[feature for feature in df_clean.columns if df_clean[feature].dtype=='object']  #Kategorik sütunları belirledik.
print("Categorical columns:",categorical_cols)


plt.figure(figsize=(15,10))

for i in range(0,len(numerical_cols)):
    plt.subplot(5,3,i+1)
    sns.kdeplot(x=df_clean[numerical_cols[i]],color='b',fill=True)
    plt.xlabel(numerical_cols[i])
    plt.tight_layout()
print(plt.show())