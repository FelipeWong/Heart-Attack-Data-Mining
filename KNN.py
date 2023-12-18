import pandas as pd
from sklearn.neighbors import NearestNeighbors


def detect_outliers(df, features, k, threshold):
    # Create a DataFrame with the selected features
    X = df[features]

    # Fit the KNN model
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(X)

    # Calculate the distances to the k nearest neighbors for each data point
    distances, indices = knn_model.kneighbors(X)

    # Calculate the average distance to the neighbors for each data point
    avg_distances = distances.mean(axis=1)

    # Identify outliers based on the distance threshold
    outliers = X[avg_distances > threshold]

    # Return the outliers
    return outliers
