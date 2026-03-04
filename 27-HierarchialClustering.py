
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# df=pd.read_csv("27-mall_Customers.csv")
#print(df.head())
# print(df.info())
# print(df.describe())

# sns.scatterplot(data=df,x="Annual Income (k$)",y="Spending Score (1-100)")
# plt.show()

import math
# def plot_all_histograms(df):
#     num_cols = df.select_dtypes(include=[np.number]).columns
#     n_cols = 3
    
    
#     n_rows = math.ceil(len(num_cols) / n_cols)
#     plt.figure(figsize=(5 * n_cols, 4 * n_rows))

#     for i, col in enumerate(num_cols, 1):
#         plt.subplot(n_rows, n_cols, i)
#         sns.histplot(df[col], bins=10,kde=True)
#         plt.title(f'Histogram of {col}')
        
#     plt.tight_layout()
#     plt.show()
# plot_all_histograms(df)

#print(df["Gender"].value_counts())

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# le=LabelEncoder()
# df["Gender"]=le.fit_transform(df["Gender"])
#print(df.head())
# df=df.drop("CustomerID",axis=1)

# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler()
# scaled_data=pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# print(scaled_data)
# plot_all_histograms(scaled_data)

import scipy.cluster.hierarchy as sch

# plt.figure(figsize=(10, 7))
# plt.title("Dendrograms")
# plt.xlabel("Customers")
# plt.ylabel("Euclidean distances")
# dendogram=sch.dendrogram(sch.linkage(scaled_data,method="ward"))
# plt.show()


# from sklearn.cluster import AgglomerativeClustering
# hc=AgglomerativeClustering(n_clusters=4)
# y_hc=hc.fit_predict(scaled_data)

# scaled_data["cluster"]=pd.DataFrame(y_hc)
# print(scaled_data.head())

# sns.scatterplot(data=scaled_data,x="Annual Income (k$)",y="Spending Score (1-100)",hue="cluster",palette="Set2")
# plt.title("Hierarchical Clustering")
#plt.show()

# from sklearn.metrics import silhouette_score
# score=silhouette_score(scaled_data,y_hc)
# print("Silhouette Score:",score)

# X=scaled_data[["Annual Income (k$)","Spending Score (1-100)"]].copy()
#print(X.head())
# hc=AgglomerativeClustering(n_clusters=4)
# y_hc=hc.fit_predict(X)

# plt.title("Customer Cluster")
# plt.show()
# print("Silhouette Score:",silhouette_score(X,y_hc))

from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score, silhouette_score
df=pd.read_csv("27-mall_Customers.csv")
df=df.drop("CustomerID",axis=1)
le=LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])

features_2d=df[["Annual Income (k$)","Spending Score (1-100)"]]
features_3d=df[["Annual Income (k$)","Spending Score (1-100)","Age"]]
features_4d=df[["Annual Income (k$)","Spending Score (1-100)","Age","Gender"]]

for feats in [features_2d,features_3d,features_4d]:
    X=feats
    X_scaled=MinMaxScaler().fit_transform(X)
    hc=AgglomerativeClustering(n_clusters=5)
    y_hc=hc.fit_predict(X_scaled)

    sil=silhouette_score(X_scaled,y_hc)
    db=davies_bouldin_score(X_scaled,y_hc)
    ch=calinski_harabasz_score(X_scaled,y_hc)
    print(f"Silhouette Score: {sil:.2f}, Davies-Bouldin Score: {db:.2f}, Calinski-Harabasz Score: {ch:.2f}")
    