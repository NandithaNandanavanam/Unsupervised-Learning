# Read Fashion MNIST dataset

import util_mnist_reader
X_train, y_train = util_mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = util_mnist_reader.load_mnist('../data/fashion', kind='t10k')

# Your code goes here . . .
#K means algorithm
#-------------------------------------------------------------------------------------------------------------------------------
#Libraries
import numpy as np
import util_mnist_reader
import matplotlib.pyplot as plt
import datetime
from sklearn import metrics, cluster
from scipy.stats import mode

#Preprocessing 
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
X = X.reshape(len(X),-1)
X = X.astype(float) / 255.0

score = []
inertia = []
centroids = []
for k in range(9, 12):
    print(k)
    #Computing the Kmeans for given data
    kmeans = cluster.KMeans(n_clusters = k)
    clusters = kmeans.fit_predict(X)
    inertia.append(kmeans.inertia_)

    #Mapping the learnt labels to true labels
    labels = np.zeros_like(clusters)
    for i in range(k):
        mask = (clusters == i)
        labels[mask] = mode(y[mask])[0]

    #Computing the accuracy for the given k value
    accuracy_score = metrics.accuracy_score(y, labels)
    print('Accuracy is', accuracy_score)
    score.append(accuracy_score)

    #confusion matrix 
    confusion_matrix = metrics.confusion_matrix(y, labels)
    print(confusion_matrix)
    
#Plot Elbow graph
print('Elbow graph')
plt.plot(range(9,12), inertia)
plt.xlabel('Number of clusters')
plt.show()
