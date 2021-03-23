# importing libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
# create data
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=0)

# create model
KM = KMeans(n_clusters=4)

# train model
KM.fit(X)

# predict
y_pred = KM.predict(X)
print(y_pred[:10])
print(y_true[:10])

# get report 
my_accuracy = accuracy_score(y_pred, y_true)
print(my_accuracy)