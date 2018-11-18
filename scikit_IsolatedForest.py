import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolatedForest:
    def __init__(self,n_features):
        rng = np.random.RandomState(42)
        self.clf = IsolationForest(max_samples=105,max_features=n_features, random_state=rng, contamination=0, behaviour="new")
        self.scaler = StandardScaler()

    def train(self,train):
        #train = self.scaler.fit_transform(train) 
        self.clf.fit(train)

    def predict(self,valid,anomaly):
        #valid = self.scaler.fit_transform(valid) 
        #anomaly = self.scaler.fit_transform(anomaly) 
        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)
        score = self.clf.decision_function(valid)
        
        print("IsolatedForest result")
        print(" Validation data:", list(y_pred_valid).count(1)/y_pred_valid.shape[0])
        print(" Outlier data:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
