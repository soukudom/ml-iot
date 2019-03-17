#!/usr/bin/env python3

import imp
from pull_data import Pull
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np

from scikit_IsolatedForest import IsolatedForest
from scikit_LOFNovelty import LOFNovelty
from scikit_OneClassSVM import OCSVM

IKEA_APP = "dev-annotated-datasets/ikea-app"
IKEA_HOMEKIT = "dev-annotated-datasets/ikea-homekit"
IP_CAM = "dev-annotated-datasets/ipcam"
NORMAL_USER = "dev-annotated-datasets/normal-user"
VOICE_ASSISTANT = "dev-annotated-datasets/voice-assistant"

t = Pull(IKEA_HOMEKIT+"/train/",1)
a = Pull(IKEA_HOMEKIT+"/anomaly/",1)
#v = Pull(IKEA_APP+"/valid/",1)

#i = IsolatedForest(t.features_cnt)
#i.train(t.data)
#i.predict(v.data,a.data)
#clf = i.getModel()
#y_pred_valid = clf.predict(v.data)
#print(y_pred_valid)


l = LOFNovelty()
l.train(t.data)
#l.predict(v.data,a.data)


#o = OCSVM()
#o.train(t.data)
#o.predict(v.data,a.data)
