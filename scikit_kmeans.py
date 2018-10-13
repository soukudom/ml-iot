import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.cluster import KMeans

class KMeansCluster:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.clf = KMeans(random_state=rng,n_clusters=2)

    def train(self,train):
        self.clf.fit(train)
        print("KMeans train clusters")
        print("labels",self.clf.labels_)

    def predict(self,valid,anomaly):
        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)
        print(" KMeans predicted clusters")
        print(" Validation cluster:",y_pred_valid)

