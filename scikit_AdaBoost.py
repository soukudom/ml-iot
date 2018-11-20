import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics


class AdaBoost:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.clf = AdaBoostClassifier(random_state=rng)

    def train(self,train,labels):
        self.clf.fit(train,labels)

    def predict(self,valid_data, valid_labels):
        y_pred = self.clf.predict(valid_data)
        print(metrics.accuracy_score(y_pred,valid_labels))
        # return value is number of predicted label
