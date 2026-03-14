# ==========================================
# M3 CLUSTERING ANALYSIS
# Dataset: spotify_complete.csv
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

# ==========================================
# 1 LOAD DATASET
# ==========================================

df = pd.read_csv("spotify_complete.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ==========================================
# 2 SELECT RAW AUDIO FEATURES
# ==========================================

features = [
    "danceability",
    "energy",
    "acousticness",
    "valence",
    "tempo",
    "loudness",
    "speechiness",
    "liveness",
    "duration_ms"
]

data = df[features].copy()

# ==========================================
# 3 CREATE DERIVED FEATURE
# acoustic_electronic_ratio
# ==========================================

data["acoustic_electronic_ratio"] = (
    df["acousticness"] / (df["energy"] + 1e-6)
)

# ==========================================
# 4 LOG TRANSFORM (fix extreme outliers)
# ==========================================

data["acoustic_electronic_ratio"] = np.log1p(
    data["acoustic_electronic_ratio"]
)

# ==========================================
# 5 STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# ==========================================
# 6 FIND BEST NUMBER OF CLUSTERS
# ==========================================

k_range = range(2, 7)

sil_scores = []
db_scores = []

for k in k_range:

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    sil_scores.append(sil)
    db_scores.append(db)

print("Silhouette scores:", sil_scores)
print("Davies-Bouldin scores:", db_scores)

# ==========================================
# 7 PLOT CLUSTER EVALUATION
# ==========================================

plt.figure()
plt.plot(k_range, sil_scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.savefig("silhouette_plot.png")
plt.show()

plt.figure()
plt.plot(k_range, db_scores, marker='o')
plt.title("Davies-Bouldin Score vs Number of Clusters")
plt.xlabel("K")
plt.ylabel("Davies-Bouldin Index")
plt.savefig("davies_bouldin_plot.png")
plt.show()

# ==========================================
# 8 FINAL KMEANS MODEL
# ==========================================

kmeans = KMeans(n_clusters=2, random_state=42)

clusters = kmeans.fit_predict(X_scaled)

data["cluster"] = clusters

print("\nCluster counts:")
print(data["cluster"].value_counts())

# ==========================================
# 9 PCA VISUALIZATION
# ==========================================

pca = PCA(n_components=2)

pca_result = pca.fit_transform(X_scaled)

data["pca1"] = pca_result[:, 0]
data["pca2"] = pca_result[:, 1]

plt.figure()

sns.scatterplot(
    x="pca1",
    y="pca2",
    hue="cluster",
    data=data,
    palette="Set2"
)

plt.title("Cluster Visualization using PCA")
plt.savefig("cluster_pca_plot.png")
plt.show()

# ==========================================
# 10 CLUSTER PROFILE
# ==========================================

cluster_profile = data.groupby("cluster").mean()

print("\nCluster Feature Averages:")
print(cluster_profile)

# ==========================================
# 11 DEDUPLICATION EXPERIMENT
# ==========================================

print("\nRunning deduplication experiment...")

if "track_id" in df.columns:
    df_dedup = df.drop_duplicates(subset="track_id")
else:
    df_dedup = df.drop_duplicates()

data_dedup = df_dedup[features].copy()

data_dedup["acoustic_electronic_ratio"] = (
    df_dedup["acousticness"] / (df_dedup["energy"] + 1e-6)
)

data_dedup["acoustic_electronic_ratio"] = np.log1p(
    data_dedup["acoustic_electronic_ratio"]
)

X_dedup = scaler.fit_transform(data_dedup)

model_dedup = KMeans(n_clusters=2, random_state=42)
labels_dedup = model_dedup.fit_predict(X_dedup)

sil_dedup = silhouette_score(X_dedup, labels_dedup)
db_dedup = davies_bouldin_score(X_dedup, labels_dedup)

print("\nOriginal Dataset:")
print("Silhouette:", silhouette_score(X_scaled, clusters))
print("Davies-Bouldin:", davies_bouldin_score(X_scaled, clusters))

print("\nDeduplicated Dataset:")
print("Silhouette:", sil_dedup)
print("Davies-Bouldin:", db_dedup)

# ==========================================
# END
# ==========================================

import csv
import random
from datetime import datetime, timedelta

def generate_large_csv(filename="dataset_32k.csv", num_rows=32000):
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "IT"]
    start_date = datetime(2020, 1, 1)

    # Open the file in write mode
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["Employee_ID", "Department", "Age", "Salary", "Join_Date", "Performance_Score"])

        # Generate and write the data rows
        for i in range(1, num_rows + 1):
            emp_id = f"EMP-{i:05d}"
            dept = random.choice(departments)
            age = random.randint(22, 60)
            salary = random.randint(40000, 150000)

            # Generate a random date within the last few years
            random_days = random.randint(0, 1500)
            join_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

            performance = round(random.uniform(1.0, 5.0), 1)

            writer.writerow([emp_id, dept, age, salary, join_date, performance])

    print(f"Successfully generated {filename} with {num_rows} rows.")

if __name__ == "__main__":
    generate_large_csv()
