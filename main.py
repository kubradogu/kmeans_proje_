import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Veri setini oku
df = pd.read_csv("Mall_Customers.csv")

print("İlk 5 satır:")
print(df.head())

# Özellik seçimi
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow yöntemi
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Küme Sayısı")
plt.ylabel("WCSS")
plt.show()

# Silhouette score hesaplama
print("\nSilhouette Score Sonuçları:")
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k} için silhouette score: {score:.4f}")

# Model kurma
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Cluster ekleme
df["Cluster"] = y_kmeans

# Ortalama değerler
print("\nKüme Ortalamaları:")
print(df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2))

# 2D Grafik
plt.figure(figsize=(8, 6))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"], s=60)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("2D K-Means Müşteri Segmentasyonu")
plt.show()

# 3D Grafik
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df["Age"],
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    s=60
)

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income")
ax.set_zlabel("Spending Score")
ax.set_title("3D K-Means Müşteri Segmentasyonu")

plt.show()