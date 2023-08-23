from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def perform_clustering(data, num_clusters=3):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Age', 'Annual Salary']])

    kmeans = KMeans(n_clusters=num_clusters)
    data['Cluster'] = kmeans.fit_predict(scaled_data)
