# 🚖 Uber Trips Analysis (Machine Learning)

## 📌 Overview
This project analyzes Uber trip data to identify **High-Demand Hotspots** using Machine Learning. It utilizes the **K-Means Clustering** algorithm from Scikit-Learn to group trip coordinates and find the optimal locations for drivers to wait.

## 🛠 Tech Stack
* **Python 3.x**
* **Scikit-learn:** K-Means Clustering (Unsupervised Learning).
* **Pandas & NumPy:** Data Manipulation.
* **Matplotlib & Seaborn:** Visualization.

## ⚙️ How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the analysis:
    ```bash
    python uber_analysis.py
    ```

## 📊 Features
* **Auto-Data Generation:** Creates realistic synthetic lat/lon data if no CSV is found.
* **Hotspot Detection:** Automatically finds the top 4 central points of high activity.
* **Visualization:** Saves a map (`uber_hotspots.png`) showing trips and the identified centroids.