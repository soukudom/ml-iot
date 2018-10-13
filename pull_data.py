from data_parser import DataParser
import os

class Pull:
    def __init__(self,idir):
        self.data = []
        self.labels = []
        self.flow_cnt = 0
        self.features_cnt = 0

        self.load_data(idir)

    def load_data(self, idir):

        # loop via all files in the source dir
        files = os.listdir(idir)
        for f in files:
            try:
                dParse = DataParser(idir + "/" + f)
                self.flow_cnt += dParse.lines_cnt
            except:
                print("Error: failued to parse file", (idir + f))
                continue

            # binary classification label
            # 1 - correct traffic
            # 0 - anomaly traffic
            label = 1 #int(f.split("-")[-1].split(".")[0])

            #tmpTLS = dParse.getTLSInfo()
            tmpBD = dParse.getByteDistribution()
            tmpIPT = dParse.getIndividualFlowIPTs()
            tmpPL = dParse.getIndividualFlowPacketLengths()
            tmp = dParse.getIndividualFlowMetadata()


            if tmp != None and tmpPL != None and tmpIPT != None:
                # iterate over every flow
                for i in range(len(tmp)):
                    tmp_data = []
                    tmp_data.extend(tmp[i])
                    tmp_data.extend(tmpPL[i])
                    tmp_data.extend(tmpIPT[i])
                    #tmp_data.extend(tmpBD[i])
                    
                   # print("FlowMetadata",tmp_data[:7])
                   # print("PacketLenghts",tmpPL[i])
                   # print("IndividualFlowIPT",tmpIPT[i])
                    #print("BD",len(tmp_data[-256:]))
                 #   tmp_data.extend(tmpTLS[i])
    
                    if self.features_cnt == 0:
                        self.features_cnt = len(tmp_data)
                    
                    self.data.append(tmp_data)
                    self.labels.append(label)

