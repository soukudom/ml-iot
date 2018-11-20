import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class RandomForest:
    def __init__(self,features_cnt):
        rng = np.random.RandomState(42)
        self.clf = RandomForestClassifier(random_state=rng,n_estimators=10)
        self.clf.n_features = features_cnt

    def train(self,train,labels):
        self.clf.fit(train,labels)

    def predict(self,valid_data,valid_labels):
        y_pred = self.clf.predict(valid_data)
        print(metrics.accuracy_score(y_pred,valid_labels))
        #print(valid_labels) 
        #y_pred_outliers = self.clf.predict(anomaly)
        # return value is number of predicted label
