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
rand = random.randrange(0,101)
X_train, X_test, y_train, y_test = train_test_split(x, y_true, test_size = 0.4, random_state= rand)
k = KNeighborsClassifier()
k.fit(X_train, y_train)
predicted = k.predict(X_test)
print(accuracy_score(y_test,predicted))


#TODO draw a confusion matrix

print(confusion_matrix (y_test,predicted))
