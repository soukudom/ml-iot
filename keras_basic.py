import json
import numpy as np
import struct, socket
from sklearn import preprocessing
from data_parser import DataParser

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

class KerasClassifier:
    def __init__(self,n_features):
        self.model = Sequential()
        self.n_features = n_features

    def train(self, data, labels):
        data_train = np.array(data)
        #for i in data_train:
        #    for j in i:
        #        print(type(j))
        #self.model = Sequential()
        self.model.add(Dense(32, activation='relu',input_dim=self.n_features))
        self.model.add(Dense(16, activation='relu',input_dim=self.n_features))
        self.model.add(Dense(8, activation='relu',input_dim=self.n_features))
        self.model.add(Dense(1,activation='softmax'))
        self.model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
        score = self.model.fit(data_train, labels, epochs=32, batch_size=10)

    def predict(self, data):
        prediction = self.model.predict(np.array(data),steps=1)

        print("Prediction result")
        ok = 0
        non_ok = 0
        for i in prediction:
            if i[0] < 0.9: 
                print(i)
                non_ok += 1
            else:
                ok += 1

        print("ok:",ok,"non-ok:",non_ok)


