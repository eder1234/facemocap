import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import seqpc

def custom_distance_metric(X, Y, missing_value_weight=1.0):
    nan_euclidean_dist = pairwise_distances(X, Y, metric='nan_euclidean')
    X_nan = np.isnan(X)
    Y_nan = np.isnan(Y)
    non_missing_values = (~X_nan[:, np.newaxis] * ~Y_nan).sum(axis=2)
    return nan_euclidean_dist / (non_missing_values + missing_value_weight)

facemocap_df = pd.read_pickle('facemocap_df.pkl')
matrices = []
# Iterate over the rows of facemocap_df and store the values of 'Original SPC' in list_arrays if 'Pathology' is False, 'Repetitive' is False and 'Movement' is 1,2,3,4 or 5
for index, row in facemocap_df.iterrows():
    if (row['Pathology'] == False) & (row['Repetitive'] == False) & (row['Movement'] in [1,2,3,4,5]):
        original_spc = row['Original SPC']
        file_name = row['File name']
        spc, spc_array = seqpc.get_spc_from_df(facemocap_df, file_name, scaled=True, interpolated=True, dental_support_frame=False, target_length=100)
        matrices.append(spc_array[:, 3:-4, :].reshape(100, 303))

imputer = SimpleImputer(strategy='mean')
imputed_matrices = [imputer.fit_transform(matrix) for matrix in matrices]

scaler = StandardScaler()
scaled_matrices = [scaler.fit_transform(matrix) for matrix in imputed_matrices]

eps = 0.5
min_samples = 5

# Apply DBSCAN to each matrix separately
cluster_results = []
for matrix in scaled_matrices:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=custom_distance_metric)
    clusters = dbscan.fit_predict(matrix)
    cluster_results.append(clusters)

# Analyze the clustering results for each matrix
for i, clusters in enumerate(cluster_results):
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    print(f"Matrix {i+1}:")
    print(f"Number of clusters: {len(unique_clusters)}")
    print(f"Cluster sizes: {counts}\n")