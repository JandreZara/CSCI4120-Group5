import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from scipy.stats import mode


x, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#TODO determine the best k for k-means 

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
visualizer.fit(x)
visualizer.show()


#TODO calculate accuracy for best k

clusters = visualizer.predict(x)
accuracy = []
labels = np.zeros_like(clusters)
for i in range(10):

    mask = (clusters == i)
    labels[mask] = mode(clusters[mask])[0]
    accuracy.append(accuracy_score(clusters , labels)) 

print('\n Accuracy per KMean: ' + repr(accuracy))

#TODO draw a confusion matrix
mat = confusion_matrix (clusters , labels)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()

print("\n")
print(confusion_matrix (clusters , labels))


