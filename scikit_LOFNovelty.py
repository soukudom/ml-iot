import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class LOFNovelty:
    def __init__(self):
        self.clf = LocalOutlierFactor(novelty=True, contamination=0.1)
        self.scaler = StandardScaler()

    def train(self,train):
        #train = self.scaler.fit_transform(train)
        self.clf.fit(train)

    def predict(self,valid,anomaly):
        #valid = self.scaler.fit_transform(valid) 
        #anomaly = self.scaler.fit_transform(anomaly)

        y_pred_valid = self.clf.predict(valid)
        y_pred_outliers = self.clf.predict(anomaly)
        score_valid = self.clf.decision_function(valid)
        score_anomaly = self.clf.decision_function(anomaly)

        print("LOF Novelty result")
        print(" Validation data:", list(y_pred_valid).count(1)/y_pred_valid.shape[0])
        #print("Score", score_valid.mean(), score_valid.std())
        print(" Outlier data:", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
        #print("Score", score_anomaly.max(), score_anomaly.std())

