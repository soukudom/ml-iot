#!/usr/bin/env python3

import imp
from pull_data import Pull
from sklearn.model_selection import KFold

# TODO input param ....
# TODO call ml backend

#t = Pull("input-data-sets/train/",1)
#a = Pull("input-data-sets/test/anomaly-traffic",1)
#v = Pull("input-data-sets/valid/",1)
c_ikea = Pull("input-data-sets/ikea/",1)
c_cam = Pull("input-data-sets/ipcam/",2)
c_data = c_ikea.data + c_cam.data
c_target = c_ikea.labels + c_cam.labels

#from keras_basic import KerasClassifier
#m = KerasClassifier(t.features_cnt)
#m.train(t.data,t.labels)
#m.predict(a.data)

#from scikit_IsolatedForest import IsolatedForest
#i = IsolatedForest(t.features_cnt)
#i.train(t.data)
#i.predict(v.data,a.data)

#from scikit_LOFNovelty import LOFNovelty
#l = LOFNovelty()
#l.train(t.data)
#l.predict(v.data,a.data)

#from scikit_RandomForest import RandomForest
#r = RandomForest(t.features_cnt)
#r.train(t.data,t.labels)
#r.predict(v.data,a.data)

#from scikit_AdaBoost import AdaBoost
#r = AdaBoost()
#r.train(t.data,t.labels)
#r.predict(v.data,a.data)

# Warninig: Minimum number of classes is 2
#from scikit_GradientBoostingClassifier import GradientBoosting
#g = GradientBoosting()
#g.train(t.data,t.labels)
#g.predict(v.data,a.data)

#from scikit_kmeans import KMeansCluster
#k = KMeansCluster()
#k.train(t.data)
#k.predict(v.data,a.data)

#from scikit_OneClassSVM import OCSVM
#o = OCSVM()
#o.train(t.data)
#o.predict(v.data,a.data)

kf = KFold(3, True, 1)
for train_index, test_index in kf.split(c_data):
    print("TRAIN:", train_index, "TEST:", test_index)
    
