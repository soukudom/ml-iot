import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.clf = AdaBoostClassifier(random_state=rng)

    def train(self,train,labels):
        self.clf.fit(train,labels)

    def predict(self,valid,anomaly):
        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)
        # return value is number of predicted label
