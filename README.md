# Amazon-Music-Clustering
**Problem Statement:**

With millions of songs available on platforms like Amazon, manually categorizing tracks into genres is impractical. 

The goal of this project is to automatically group similar songs based on their audio characteristics such as tempo, energy, danceability, etc., using clustering techniques.

This helps in Playlist curation.

**Skills TakeAway from this Project:**

Data Exploration & Cleaning

Feature Engineering & Data Normalization

K-Means Clustering & Elbow Method

DBSCAN, and Agglomerative Clustering

Cluster Evaluation & Comparison (Silhouette Score and Davies–Bouldin Index)

Dimensionality Reduction (PCA)

Cluster Visualization & Interpretation

Python for Machine Learning (Pandas, Numpy, Scikit learn, Matplotlib, Seaborn)

Streamlit

Data Storytelling & Insight Communication

**⚙️ Tech Stack**

Python 3.10+

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, joblib

IDE: Jupyter Notebook / VS Code

Visualization: Streamlit Dashboard

**Import Libraries:**

** === Data Handling ===**

import pandas as pd

import numpy as np

** === Visualization ===**

import matplotlib.pyplot as plt

import seaborn as sns

**=== Machine Learning & Clustering ===**

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from sklearn.metrics import silhouette_score, davies_bouldin_score

**Approach:**

Data Loading & Exploration – Imported the Amazon Music dataset and explored song-level attributes like energy, tempo, and valence using pandas and seaborn.

Data Cleaning & Feature Engineering – Removed irrelevant columns, handled missing values, and converted duration_ms to duration_min for better interpretability.

Feature Scaling & Transformation – Normalized all numeric features using StandardScaler to prepare for clustering.

Dimensionality Reduction (PCA) – Reduced high-dimensional data into 2 principal components to visualize patterns and variance among songs.

Clustering Model Development – Applied KMeans, DBSCAN, and Agglomerative Clustering to group songs

Cluster Evaluation & Profiling – Compared models using Silhouette Score and Davies–Bouldin Index

Visualization & Insights – Visualized clusters through PCA scatter plots and heatmaps; extracted feature-wise summaries for each cluster.

Dashboard Development – Built an interactive Streamlit app to explore clusters dynamically, visualize feature comparisons, and display top 10 songs per cluster.

**Snapshot:**

<img width="1918" height="1006" alt="image" src="https://github.com/user-attachments/assets/6920330e-f395-46f6-b6d3-c997e95a1f47" />

<img width="1918" height="905" alt="image" src="https://github.com/user-attachments/assets/c093546a-5f96-4025-9f92-7e1c7ba522b5" />

