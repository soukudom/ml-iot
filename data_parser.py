import numpy as np
import math
import json
import gzip



class DataParser:
    def __init__(self, json_file, analyse, compact):
        self.flows = []
        self.compact = compact
        self.analyse = analyse
        self.lines_cnt = 0
        self.error_cnt = 0
        self.error = None

        with open(json_file,'r') as fp:
            for line in fp:
                try:
                    tmp = json.loads(line)
                    self.flows.append(tmp)
                    self.lines_cnt += 1
                except Exception as e:
                    self.error = e
                    self.error_cnt += 1
                    continue

#    def getTLSInfo(self):
#        if self.flows == []:
#            return None

#        data = []
#        for flow in self.flows:
#            if len(flow['packets']) == 0:
#                continue
#            tls_info = np.zeros(len(cs.keys())+len(ext.keys())+1)
#
#            if 'tls' in flow and 'cs' in flow['tls']:
#                for c in flow['tls']['cs']:
#                    if c in cs:
#                        tls_info[cs[c]] = 1
#            else:
#                data.append([])
#                continue
#
#            if 'tls' in flow and 'c_extensions' in flow['tls']:
#                for c in flow['tls']['c_extensions']:
#                    if c.keys()[0] in ext:
#                        tls_info[len(cs.keys())+ext[c.keys()[0]]] = 1
#
#            if 'tls' in flow and 'c_key_length' in flow['tls']:
#                tls_info[len(cs.keys())+len(ext.keys())] = flow['tls']['c_key_length']
#
#            data.append(list(tls_info))
#
#        return data
        

    def getByteDistribution(self):
        if self.flows == []:
            return None

        data = []
        data2 = []
        for flow in self.flows:
            #if len(flow['packets']) == 0:
            #    continue
            #print("data parser len and sum",len( flow["byte_dist"]),sum( flow["byte_dist"]))
            # Divide every item in the field by sum of items
            if 'byte_dist' in flow and sum(flow['byte_dist']) > 0:
                tmp = map(lambda x: x/float(sum(flow['byte_dist'])),flow['byte_dist'])
                #data.append(tmp)
                data2.append(list(tmp))
            else:
                #data.append(np.zeros(256))
                data2.append(np.zeros(256))
        return data, data2


    def getIndividualFlowPacketLengths(self):
        if self.flows == []:
            return None

        data = []
        analyse_data = [] #list of matrixes for jupyter graphs

        if self.compact:
            numRows = 10
            binSize = 150.0
        else:
            numRows = 60
            binSize = 25.0
        for flow in self.flows:
            transMat = np.zeros((numRows,numRows))
            if len(flow['packets']) == 0:
                data.append(list(transMat.flatten()))
                analyse_data.append(list(transMat.flatten()))
                continue
            # Just one packet in the flow
            elif len(flow['packets']) == 1:
                curPacketSize = min(int(flow['packets'][0]['b']/binSize),numRows-1)
                transMat[curPacketSize,curPacketSize] = 1
                data.append(list(transMat.flatten()))
                analyse_data.append(list(transMat.flatten()))
                continue

            # get raw transition counts
            # Fill in matrix based on number of bytes in one packet
            for i in range(1,len(flow['packets'])):
                prevPacketSize = min(int(flow['packets'][i-1]['b']/binSize),numRows-1)
                if 'b' not in flow['packets'][i]:
                    break
                curPacketSize = min(int(flow['packets'][i]['b']/binSize),numRows-1)
                transMat[prevPacketSize,curPacketSize] += 1

            if self.analyse == 1:
                analyse_data.append(list(transMat.flatten()))
           
            else: 
                # get empirical transition probabilities
                # Divide every row by its sum of items
                for i in range(numRows):
                    if float(np.sum(transMat[i:i+1])) != 0:
                        transMat[i:i+1] = transMat[i:i+1]/float(np.sum(transMat[i:i+1]))

                data.append(list(transMat.flatten()))

        if self.analyse == 1:
            return analyse_data
        else:
            return data


    def getIndividualFlowIPTs(self):
        if self.flows == []:
            return None

        data = []
        analyse_data = [] #list of matrixes for jupyter graphs
        if self.compact:
            numRows = 10
            binSize = 50.0
        else:
            numRows = 30
            binSize = 50.0
        # Similar to the getIndividualFlowPacketsLengths
        for flow in self.flows:
            transMat = np.zeros((numRows,numRows))
            if len(flow['packets']) == 0:
                data.append(list(transMat.flatten()))
                analyse_data.append(list(transMat.flatten()))
                continue
            elif len(flow['packets']) == 1:
                curIPT = min(int(flow['packets'][0]['ipt']/float(binSize)),numRows-1)
                transMat[curIPT,curIPT] = 1
                data.append(list(transMat.flatten()))
                analyse_data.append(list(transMat.flatten()))
                continue

            # get raw transition counts
            for i in range(1,len(flow['packets'])):
                prevIPT = min(int(flow['packets'][i-1]['ipt']/float(binSize)),numRows-1)
                curIPT = min(int(flow['packets'][i]['ipt']/float(binSize)),numRows-1)
                transMat[prevIPT,curIPT] += 1
                
            if self.analyse == 1:
                analyse_data.append(list(transMat.flatten()))
            else:
                # get empirical transition probabilities
                for i in range(numRows):
                    if float(np.sum(transMat[i:i+1])) != 0:
                        transMat[i:i+1] = transMat[i:i+1]/float(np.sum(transMat[i:i+1]))

                data.append(list(transMat.flatten()))

        if self.analyse == 1:
            return analyse_data
        else:    
            return data


    def getIndividualFlowMetadata(self, PKTS, BYTES, FLOW_TIME, WHT, BYTE_DIST_M, BYTE_DIST_S, ENTROPY, IDP):
        if self.flows == []:
            return None

        data = []
        for flow in self.flows:
            #if len(flow['packets']) == 0:
            #    continue
            tmp = []

            if PKTS or self.analyse:
                if self.analyse:
                    # Pkts in
                    if 'num_pkts_in' in flow:
                        tmp.append(flow['num_pkts_in']) # inbound packets
                    else:
                        tmp.append(0)
                    # Pkts out
                    if 'num_pkts_out' in flow:
                        tmp.append(flow['num_pkts_out']) # outbound packets
                    else:
                        tmp.append(0)
                else:
                    if 'num_pkts_in' in flow and 'num_pkts_out' in flow:
                        if flow['num_pkts_out'] == 0 and flow['num_pkts_in'] == 0:
                            tmp.append(0)
                        elif flow['num_pkts_out'] == 0:
                            tmp.append(flow['num_pkts_in'])
                        elif flow['num_pkts_in'] == 0:
                            tmp.append(1/flow['num_pkts_out'])
                        else:
                            tmp.append(flow['num_pkts_in']/flow['num_pkts_out'])
                    elif 'num_pkts_in' in flow:
                        tmp.append(flow['num_pkts_in'])
                    else:
                        if flow['num_pkts_out'] == 0:
                            tmp.append(0)
                        else:   
                            tmp.append(1/flow['num_pkts_out'])
            if BYTES or self.analyse:
                if self.analyse:
                    # Bytes in
                    if 'bytes_in' in flow:
                        tmp.append(flow['bytes_in']) # inbound bytes
                    else:
                        tmp.append(0)
                    # Bytes out
                    if 'bytes_out' in flow:
                        tmp.append(flow['bytes_out']) # outbound bytes
                    else:
                        tmp.append(0)
                else: 
                    # Bytes in/Bytes out Ration
                    if 'bytes_in' in flow and 'bytes_out' in flow:
                        if flow['bytes_in'] == 0 and flow['bytes_out'] == 0:
                            tmp.append(0)
                        elif flow['bytes_in'] == 0:
                            tmp.append(1/flow['bytes_out'])
                        elif flow['bytes_out'] == 0:
                            tmp.append(flow['bytes_in'])
                        else:
                            tmp.append(flow['bytes_in']/flow['bytes_out'])                    
                    elif 'bytes_in' in flow:
                        tmp.append(flow['bytes_in'])
                    else:
                        if flow['bytes_out'] == 0:
                            tmp.append(0)
                        else:
                            tmp.append(1/flow['bytes_out'])
                
            if FLOW_TIME or self.analyse:
                # Elapsed time of flow
                if flow['packets'] == []:
                    tmp.append(0)
                else:
                    time = 0
                    for packet in flow['packets']:
                        time += packet['ipt']
                    tmp.append(time)
            if WHT or self.analyse:
                # WHT
                if "wht" in flow:
                    whtFields = list(flow["wht"])
                    tmp.append(whtFields[0])
                    tmp.append(whtFields[1])
                    tmp.append(whtFields[2])
                    tmp.append(whtFields[3])
                else:
                    tmp.append(0)
                    tmp.append(0)
                    tmp.append(0)
                    tmp.append(0)
            if BYTE_DIST_M or self.analyse:
                # Byte Dist Mean
                if "byte_dist_mean" in flow:
                    tmp.append(flow["byte_dist_mean"])
                else:
                    tmp.append(0)
            if BYTE_DIST_S or self.analyse:
                # Byte Dist Std
                if "byte_dist_std" in flow:
                    tmp.append(flow["byte_dist_std"])
                else:
                    tmp.append(0) 
            if ENTROPY or self.analyse:
                # Entropy
                if "entropy" in flow:
                    tmp.append(flow["entropy"])
                else:
                    tmp.append(0)
            if IDP or self.analyse:
                # IDP in
                if "idp_len_in" in flow:
                    tmp.append(flow["idp_len_in"])
                else:
                    tmp.append(0)
                # IDP out
                if "idp_len_out" in flow:
                    tmp.append(flow["idp_len_out"])
                else:
                    tmp.append(0)

            data.append(tmp)

        if data == []:
            return None
        return data

