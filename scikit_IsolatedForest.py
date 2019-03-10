import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


class IsolatedForest:
    def __init__(self,n_features):
        rng = np.random.RandomState(42)
        self.clf = IsolationForest(max_samples="auto",max_features=100, random_state=rng, contamination=0)
        self.scaler = StandardScaler()

    def train(self,train):
        #train = self.scaler.fit_transform(train) 
        self.clf.fit(train)

    def getModel(self):
        return self.clf

    def predict(self,valid,anomaly):
        #valid = self.scaler.fit_transform(valid) 
        #anomaly = self.scaler.fit_transform(anomaly) 
        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)
        score = self.clf.decision_function(valid)
        
        print("IsolatedForest result")
        print(confusion_matrix([1]*len(y_pred_valid),y_pred_valid).ravel())
        print(confusion_matrix([-1]*len(y_pred_outliers),y_pred_outliers).ravel())
        #tn, fp, fn, tp = confusion_matrix([1]*len(y_pred_outliers),y_pred_outliers)
        confusion_matrix([1]*len(y_pred_valid),y_pred_valid)
        print(" Validation data:", list(y_pred_valid).count(1)/y_pred_valid.shape[0])
        print(" Outlier data:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
