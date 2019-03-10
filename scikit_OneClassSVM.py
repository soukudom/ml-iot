import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class OCSVM:
    def __init__(self):
        # random state is deprecated
        # rng = np.random.RandomState(42)
        self.clf = OneClassSVM(gamma="scale", nu=0.1)
        self.scaler = StandardScaler()

    def train(self,train):
        #train = self.scaler.fit_transform(train)
        self.clf.fit(train)

    def predict(self,valid,anomaly):
        #valid = self.scaler.fit_transform(valid) 
        #anomaly = self.scaler.fit_transform(anomaly)

        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)

        print("OneClassSVM result")
        print(confusion_matrix([1]*len(y_pred_valid),y_pred_valid).ravel())
        print(confusion_matrix([-1]*len(y_pred_outliers),y_pred_outliers).ravel())
        print(" Validation data:", list(y_pred_valid).count(1)/y_pred_valid.shape[0])
        print(" Outlier data:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
