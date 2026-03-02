import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("26-customer_data.csv")
#print(df.head())

#sns.scatterplot(data=df,x="Annual_Income",y="Spending_Score")
#plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test=train_test_split(df,test_size=0.2,random_state=15)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Standard scaler ile de yapsak aynı sonucu verecektir çünkü K-Means algoritması ölçeklendirmeye duyarlıdır.

from sklearn.cluster import KMeans

#elbow method
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++") #init default değeri k-means++'dır, bu yöntem rastgele merkez seçimine göre daha iyi sonuç verir.
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)
#print(wcss)

# plt.plot(range(1,11),wcss)
# plt.title("Elbow Method")
# plt.xlabel("Number of Clusters")
# plt.xticks(range(1,11))
# plt.ylabel("WCSS")
# plt.show()

kmeans=KMeans(n_clusters=3)
kmeans.fit(X_train_scaled)

y_pred=kmeans.predict(X_test_scaled)
# sns.scatterplot(data=pd.DataFrame(X_test_scaled,columns=X_test.columns),x="Annual_Income",y="Spending_Score",hue=y_pred)
# plt.show()

#Other alternatives to find the optimal number of clusters

from kneed import KneeLocator
k1=KneeLocator(range(1,11),wcss,curve="convex",direction="decreasing")
#print("Optimal number of clusters:",k1.elbow)

#silhouette method
# from sklearn.metrics import silhouette_score
# silhouette_scores=[]
# for i in range(2,11):
#     kmeans=KMeans(n_clusters=i)
#     kmeans.fit(X_train_scaled)
#     score=silhouette_score(X_train_scaled,kmeans.labels_)
#     silhouette_scores.append(score)
# plt.plot(range(2,11),silhouette_scores)
# plt.xticks(range(2,11))
# plt.title("Silhouette Scores")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette coefficient")
# plt.show()








