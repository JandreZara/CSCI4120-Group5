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


x, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#TODO determine the best k for k-means 

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
visualizer.fit(x)
visualizer.show()


#TODO calculate accuracy for best k
accuracy = []
for i in range(1,11):
    rand = random.randrange(0, 101)
    k = KMeans(n_clusters=i, random_state= rand)
    k.fit(x, y_true)
    predicted = k.predict(x)
    accuracy.append(accuracy_score(y_true,predicted))

print('\n Accuracy per KMean: ' + repr(accuracy))

#TODO draw a confusion matrix
mat = confusion_matrix (y_true,predicted)
sns.heatmap(mat.T, square = True, annot = True, fmt = 'd', cbar = False)
plt.show()
#plt.xLabel('True Label')
#plt.yLabel('Predicted Label')

print(confusion_matrix (y_true,predicted))
