import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self,features_cnt):
        rng = np.random.RandomState(42)
        self.clf = RandomForestClassifier(random_state=rng)
        self.clf.n_features = features_cnt

    def train(self,train,labels):
        self.clf.fit(train,labels)

    def predict(self,valid,anomaly):
        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)
        # return value is number of predicted label
