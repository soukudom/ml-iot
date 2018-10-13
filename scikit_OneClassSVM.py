import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.svm import OneClassSVM


class OCSVM:
    def __init__(self):
        # random state is deprecated
        # rng = np.random.RandomState(42)
        self.clf = OneClassSVM(gamma="scale")

    def train(self,train):
        self.clf.fit(train)

    def predict(self,valid,anomaly):
        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)

        print("OneClassSVM result")
        print(" Validation data:", list(y_pred_valid).count(1)/y_pred_valid.shape[0])
        print(" Outlier data:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])