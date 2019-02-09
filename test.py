#!/usr/bin/env python3
from data_parser import DataParser


dParse = DataParser("annotated-datasets/voice-assistant/google-mini-additional/google-mac2.json")
tmp = dParse.getIndividualFlowPacketLengths()
print(tmp[0])
print(tmp[1])
