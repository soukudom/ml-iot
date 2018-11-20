import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


class GradientBoosting:
    def __init__(self):
        rng = np.random.RandomState(42)
        self.clf = GradientBoostingClassifier(random_state=rng)

    def train(self,train,labels):
        self.clf.fit(train,labels)

    def predict(self,valid_data, valid_labels):
        y_pred = self.clf.predict(valid_data)
        print(metrics.accuracy_score(y_pred,valid_labels))
