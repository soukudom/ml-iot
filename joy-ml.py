#!/usr/bin/env python3

from pull_data import Pull
import imp

# TODO input param ....
# TODO call ml backend

t = Pull("input-data-sets/train/")
a = Pull("input-data-sets/test/anomaly-traffic")
v = Pull("input-data-sets/valid/")

#from keras_basic import KerasClassifier
#m = KerasClassifier()
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

#from scikit_RandomForest import RandomForest
#r = RandomForest(t.features_cnt)
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

from scikit_OneClassSVM import OCSVM
o = OCSVM()
o.train(t.data)
o.predict(v.data,a.data)
