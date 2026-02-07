import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

# ==========================================
# CONFIGURATION
# ==========================================
GRAPH_FILENAME = "uber_hotspots.png"
DATA_FILENAME = "uber_trips.csv"

# ==========================================
# 1. DATA GENERATION (Synthetic)
# ==========================================
def get_data(filename=DATA_FILENAME):
    """
    Loads 'uber_trips.csv' if it exists. 
    Otherwise, generates realistic synthetic Uber trip data.
    """
    if os.path.exists(filename):
        print(f"[✔] Loading real data from {filename}...")
        df = pd.read_csv(filename)
    else:
        print(f"[!] '{filename}' not found. Generating synthetic data...")
        np.random.seed(42)
        n_samples = 1000
        
        # Simulate NY/City coordinates (roughly)
        # 3 main "Hotspots" (e.g., Airport, Downtown, Suburbs)
        centers = [[40.71, -74.00], [40.75, -73.98], [40.80, -73.95]]
        
        latitudes = []
        longitudes = []
        
        for _ in range(n_samples):
            # Randomly pick one of the 3 centers
            center = centers[np.random.randint(0, 3)]
            # Add random noise (spread)
            lat = center[0] + np.random.normal(0, 0.02)
            lon = center[1] + np.random.normal(0, 0.02)
            latitudes.append(lat)
            longitudes.append(lon)
            
        # Generate random hours (peak times: 8am and 6pm)
        hours = np.random.choice(range(0, 24), size=n_samples, p=[
            0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.1, 0.15, 0.1, 0.05, # Morning
            0.04, 0.04, 0.04, 0.05, 0.05, 0.06, 0.1, 0.15, 0.1, 0.05, # Evening
            0.04, 0.03, 0.02, 0.01
        ])
        
        df = pd.DataFrame({'Lat': latitudes, 'Lon': longitudes, 'Hour': hours})
        
    return df

# ==========================================
# 2. ANALYSIS & ML CLUSTERING
# ==========================================
def analyze_hotspots(df, n_clusters=3):
    """
    Uses K-Means Clustering to find high-demand locations.
    """
    print("[-] Running K-Means Clustering to find hotspots...")
    
    # We only need Lat/Lon for clustering
    X = df[['Lat', 'Lon']]
    
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    # Fit the model
    kmeans.fit(X)
    
    # Get the cluster labels (which group each trip belongs to)
    df['Cluster'] = kmeans.labels_
    
    # Get the coordinates of the "Hotspots" (Centroids)
    centroids = kmeans.cluster_centers_
    
    return df, centroids

# ==========================================
# 3. VISUALIZATION
# ==========================================
def plot_results(df, centroids):
    """
    Plots all trips and highlights the ML-detected hotspots.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot all trips, colored by their cluster
    sns.scatterplot(
        x='Lon', y='Lat', data=df, 
        hue='Cluster', palette='viridis', 
        s=30, alpha=0.6
    )
    
    # Plot the Centroids (Hotspots)
    plt.scatter(
        centroids[:, 1], centroids[:, 0], 
        s=300, c='red', marker='X', 
        label='ML Detected Hotspot', edgecolors='black'
    )
    
    plt.title('Uber Trip Analysis: Identifying High-Demand Areas (K-Means)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Update legend to show both cluster colors and hotspot marker
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save Graph
    save_path = os.path.join(os.getcwd(), GRAPH_FILENAME)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[✔] Hotspot Map saved at: {save_path}")
    
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("==========================================")
    print("   UBER TRIPS ANALYSIS (ML CLUSTERING)")
    print("==========================================\n")

    # 1. Get Data
    df = get_data()
    print(f"[✔] Loaded {len(df)} trip records.")
    
    # 2. Basic Stats
    peak_hour = df['Hour'].mode()[0]
    print(f"[-] Peak Rush Hour: {peak_hour}:00")
    
    # 3. ML Analysis
    df, hotspots = analyze_hotspots(df, n_clusters=4)
    
    print("\n[!] Identified Top 4 Pickup Hotspots (Lat, Lon):")
    for i, spot in enumerate(hotspots):
        print(f"    {i+1}. {spot[0]:.4f}, {spot[1]:.4f}")
        
    # 4. Visualize
    print("\n[-] Generating map...")
    plot_results(df, hotspots)