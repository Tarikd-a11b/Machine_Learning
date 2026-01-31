
import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import resample
import matplotlib.pyplot as plt

#random seed for reproducibility

np.random.seed(42)

set1no=900
set2no=100

df1=pd.DataFrame({
    "feature1": np.random.normal(loc=0,scale=1,size=set1no),
    "feature2": np.random.normal(loc=0,scale=1,size=set1no),
    "target":[0 ]*set1no
})
df2=pd.DataFrame({
    "feature1": np.random.normal(loc=0,scale=1,size=set2no),
    "feature2": np.random.normal(loc=0,scale=1,size=set2no),
    "target":[1]*set2no
})

print(df1.head())
print(df2.head())

df=pd.concat([df1,df2]).reset_index (drop=True)
print(df)


df_minority=df[df['target']==1]
df_majority=df[df['target']==0]


resampled_minority= resample(
    df_minority,replace=True,
    n_samples=len(df_majority),
    random_state=42
)
print(resampled_minority)

df_majority= resample(
    df_majority,replace=False,
    n_samples=len(df_minority),
    random_state=42
)
print(df_majority)


#SMOTE (Synthetic Minority Over-sampling Technique)


print(df)
from imblearn.over_sampling import SMOTE 

oversample=SMOTE()
(x,y)=oversample.fit_resample(df[['feature1','feature2']],df['target'])
print(x)
print(y)

oversample_df=pd.concat([x,y],axis=1)
oversample_df["target"].value_counts()

plt.scatter(oversample_df['feature1'],oversample_df['feature2'],c=oversample_df['target'])
plt.show()
