#!/usr/bin/env python3
import imp
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
#from data_parser import DataParser
from pull_data import Pull
import os

IKEA_APP = "dev-annotated-datasets/ikea-app"
IKEA_HOMEKIT = "dev-annotated-datasets/ikea-homekit"
IP_CAM = "dev-annotated-datasets/ipcam"
NORMAL_USER = "dev-annotated-datasets/normal-user"
VOICE_ASSISTANT = "dev-annotated-datasets/voice-assistant"

t = Pull(IKEA_APP+"/train/",1)
