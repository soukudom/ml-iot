import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoosting:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.clf = GradientBoostingClassifier(random_state=rng)

    def train(self,train,labels):
        self.clf.fit(train,labels)

    def predict(self,valid,anomaly):
        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)

        print("GradientBoosting result")
        print(" Validation data:", list(y_pred_valid).count(1)/y_pred_valid.shape[0])
        print(" Outlier data:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
