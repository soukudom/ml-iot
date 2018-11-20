#!/usr/bin/env python3

import imp
from pull_data import Pull
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np

# TODO input param ....
# TODO call ml backend

t = Pull("input-data-sets/train/",1)
a = Pull("input-data-sets/test/anomaly-traffic",1)
v = Pull("input-data-sets/valid/",1)

c_ikea = Pull("input-data-sets/ikea/",1)
c_cam = Pull("input-data-sets/ipcam/",2)
c_data = np.array(c_ikea.data + c_cam.data)
c_target = np.array(c_ikea.labels + c_cam.labels)

#from keras_basic import KerasClassifier
#m = KerasClassifier(t.features_cnt)
#m.train(t.data,t.labels)
#m.predict(a.data)

from scikit_IsolatedForest import IsolatedForest
i = IsolatedForest(t.features_cnt)
i.train(t.data)
i.predict(v.data,a.data)

from scikit_LOFNovelty import LOFNovelty
l = LOFNovelty()
l.train(t.data)
l.predict(v.data,a.data)

#from scikit_kmeans import KMeansCluster
#k = KMeansCluster()
#k.train(t.data)
#k.predict(v.data,a.data)

from scikit_OneClassSVM import OCSVM
o = OCSVM()
o.train(t.data)
o.predict(v.data,a.data)

from scikit_RandomForest import RandomForest
#X_train, X_test, y_train, y_test = train_test_split(c_data, c_target, random_state=6)
kf = KFold(3, True, 1)
print("Random Forest")
r = RandomForest(c_ikea.features_cnt)
for train_index, test_index in kf.split(c_data,c_target):
    r.train(c_data[train_index],c_target[train_index])
    r.predict(c_data[test_index],c_target[test_index])

print("AdaBoost")
from scikit_AdaBoost import AdaBoost
a = AdaBoost()
for train_index, test_index in kf.split(c_data,c_target):
    a.train(c_data[train_index],c_target[train_index])
    a.predict(c_data[test_index],c_target[test_index])

from scikit_GradientBoostingClassifier import GradientBoosting
print("GradientBoosting")
g = GradientBoosting()
for train_index, test_index in kf.split(c_data,c_target):
    g.train(c_data[train_index],c_target[train_index])
    g.predict(c_data[test_index],c_target[test_index])
