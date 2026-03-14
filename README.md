# Spotify-Data-Analysis-and-Clustering

# 🎵 Spotify Music Clustering & Pattern Discovery (CS-4412 M3)

This repository contains the **M3: Complete Implementation** for the CS-4412 Data Mining project.
It demonstrates a full end-to-end pipeline for **Spotify track analysis**, expanding beyond M2 with clustering, classification, dimensionality reduction, and anomaly detection.

---

## 📌 Project Overview
- **Dataset**: ~8,947 tracks 19 features simulated Spotify tracks (with duplicates and ratio outliers).
- **Goal**: Discover meaningful musical patterns, clusters, and anomalies.
- **Expansion**: Implements multiple techniques beyond M2, with validation and interpretation.

---

## ⚙️ Pipeline Steps
1. **Data Generation & Cleaning**
   - Simulated realistic Spotify track features (danceability, energy, acousticness, etc.).
   - Introduced and corrected **ratio outliers** (`acoustic_electronic_ratio`).
   - Deduplicated multi-genre tracks using `track_id`.

2. **Clustering (K-Means)**
   - Compared **k=2–6** clusters using **Silhouette** and **Davies-Bouldin** scores.
   - Selected **k=2** as optimal (highest Silhouette).
   - Cluster profiles interpreted via feature averages.

3. **Analytical Expansion (M3 Requirements)**
   - **PCA**: Reduced dimensionality for visualization (31% variance explained).
   - **Decision Tree**: Interpretable rules explaining cluster separation.
   - **LOF Anomaly Detection**: Identified ~10% unusual tracks.

4. **Visualization**
   - Ratio outlier correction (histograms).
   - Clustering validation metrics.
   - PCA scatter plot of clusters.
   - Energy vs. log-ratio separation.
   - Decision Tree feature importances.
   - Anomaly energy profiles.

5. **Results**
   - Cluster characterization (distinct musical profiles).
   - Key discriminating features: `ratio_log`, `energy`, `speechiness`.
   - Anomalies show higher-than-average energy.

---

## 📊 Key Findings
- **Cluster 0**: Higher energy, lower acousticness → electronic/dance-oriented tracks.
- **Cluster 1**: Higher acousticness, lower energy → acoustic/folk-oriented tracks.
- **Top Discriminators**: `ratio_log` and `energy` dominate cluster separation.
- **Anomalies**: Edge cases with unusually high energy values.

---

## 📂 Repository Structure


├── spotify_pipeline.ipynb   # Complete analysis notebook ├── spotify_complete.csv     # Cleaned + annotated dataset ├── spotify_results.png      # Visualization summary ├── README.md                # Project documentation

---

## 🚀 How to Run
1. Clone the repository:
   Open your terminal or command prompt and run:
   ```bash
   git clone https://github.com/gourobdas0103/spotify_M3_Complete_Implementation.git
   cd spotify-clustering
2. 🐍 Set Up a Virtual Environment (Recommended)
   Create and activate a virtual environment to manage dependencies:
   ```bash
   # Create virtual environment
   python -m venv venv
   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

3. Install dependencies:
   Install all required Python libraries using the provided requirements.txt:
   ```bash
   pip install -r requirements.txt

4. Run the notebook:
   Start Jupyter Notebook and open the analysis file
   ```bash
   jupyter notebook spotify_M3_Complete_Implementation.ipynb

📚 Techniques & Libraries- Clustering: K-Means (scikit-learn)
- Classification: Decision Trees (scikit-learn)
- Dimensionality Reduction: PCA (scikit-learn)
- Anomaly Detection: Local Outlier Factor (scikit-learn)
- Visualization: Matplotlib, Seaborn
- Data Handling: Pandas, NumPy

🎯 Deliverables- End-to-end pipeline (data → results).
- Analytical expansion beyond M2.
- Substantial results with interpretation.
- Visualizations connecting findings to discovery questions.

📖 References- Zaki & Meira, Data Mining and Machine Learning
- Aggarwal, Outlier Analysis
- Scikit-learn Documentation
- PyOD Documentation

## 📊 Visualization Results

Below are the key outputs generated from the Spotify clustering pipeline.

### Clustering & Analysis Output

![Spotify Clustering Results](silhouette_plot.png) ![Spotify Clustering Results](davies_bouldin_plot.png)
![Spotify Clustering Results](cluster_pca_plot.png)

The visualization summarizes:

- PCA cluster separation
- Energy vs log-ratio distribution
- Decision Tree feature importance
- Anomaly energy comparison

These plots help interpret cluster behavior and identify unusual tracks.
