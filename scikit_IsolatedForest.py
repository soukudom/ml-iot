import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest


class IsolatedForest:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.clf = IsolationForest(max_samples=105,max_features=207, random_state=rng, contamination=0, behaviour="new")

    def train(self,train):
        self.clf.fit(train)

    def predict(self,valid,anomaly):
        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)
        
        print("IsolatedForest result")
        print(" Validation data:", list(y_pred_valid).count(1)/y_pred_valid.shape[0])
        print(" Outlier data:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])