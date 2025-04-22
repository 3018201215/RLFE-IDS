# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModel
# from modelscope import AutoTokenizer, AutoModel
import csv
import json 
import pandas as pd
import numpy as np
from numpy.linalg import norm
import random
import os
import torch
import time
import logging
import torch.nn as nn


def eucliDist(v1, v2):
    return np.sqrt(sum(np.power((v1 - v2), 2)))

def cosDist(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def mDist(x1, x2):
    x1 = torch.reshape(x1,[1024])
    x2 = torch.reshape(x2,[1024])
    data = torch.stack((x1, x2))
    # print(data.shape)
    data = torch.reshape(data,(2,1024))
    cov = torch.cov(data.t())
    # 计算差值
    delta = x1 - x2
    # print(delta.shape)
    # 计算协方差矩阵的逆
    cov_inv = torch.linalg.pinv(cov)
    # print(cov_inv.shape)
    # 计算马氏距离
    m_dist = torch.sqrt(torch.dot(delta, torch.matmul(cov_inv, delta)))
    return m_dist

def searchfile(Path, File):
    li = []
    files = os.listdir(Path)
    for f in files:
        if os.path.isdir(Path+os.sep+f):
            searchfile(Path+os.sep+f, File)
        elif f.split('.')[-1] == File:
            li.append(f)
    return li



class query:

    def __init__(self, vectorDatabase, testSet, llmModel, llmTokenizer):
        self.vectorDatabase = vectorDatabase
        self.llmModel = llmModel
        self.testSet = testSet
        self.llmTokenizer = llmTokenizer

    def findSimilarity_dist(self, vector):
        similarity = {}
        result = {}
        for item in self.vectorDatabase.keys():
            base = self.vectorDatabase[item]
            baseVector = base[:-1].astype(np.float32)
            lable = base[-1]
            if baseVector.shape[0] != 1024 :
                continue
            # print(baseVector.shape)
            # print(vector.shape)
            # print(baseVector)
            baseVector = torch.Tensor(baseVector)
            vector = torch.Tensor(vector)
            baseVector = torch.reshape(baseVector,(1,1024))
            vector = torch.reshape(vector,(1,1024))
            # print(baseVector.shape)
            # print(vector.shape)
            # print(baseVector)
            pdist = nn.PairwiseDistance(p=2)
            dist = pdist(vector, baseVector)
            # dist = mDist(vector, baseVector)
            l = [dist, lable]
            similarity[item] = l
        re = sorted(similarity.items(), key=lambda x : x[1][0])
        re = re[:3]
        for i in re:
            result[i[0]] = i[1]
        return result

    def getResult(self, prompt, logPath1, logPath2):
        log1 = open(logPath1, 'w', encoding='utf-8')
        log2 = open(logPath2, 'w', encoding='utf-8')
        # file = open(jsonPath, 'w', encoding='utf-8')
        l = len(self.testSet.keys())
        similarity_time = 0
        cost_time = 0
        llm_time = 0
        cnt = 0
        right = 0
        wrong = 0
        for item in self.testSet.keys():
            my_json = {}
            t1 = time.time()
            cnt += 1
            print("{a} / {b}, {c}%".format(a = cnt, b = l, c = int(cnt/l*100)))
            base = self.testSet[item]
            vector = base[:-1].astype(np.float32)
            if vector.shape[0] != 1024 :
                continue
            label = base[-1]
            t2 = time.time()
            # similarity = self.findSimilarity_FDeep(vector, model)
            similarity = self.findSimilarity_dist(vector)
            t3 = time.time()
            # print("Similarity spend time: {}".format(t3-t2))
            similarity_time += t3 - t2
            s = ""
            s2 = []
            for i in similarity.keys():
                s += "-Input: \n" + str(i) + "\n" + "-The class label: \n" + str(similarity[i][1]) + '\n'
                s2.append(str(similarity[i][1]))
            maxlabel = max(s2,key=s2.count)
            prompt1 = prompt.format(a=s, b=str(item))
            if cnt % 500 == 0 or cnt == 1:
                print(prompt1)
                log1.write("Prompt: " + prompt1 + '\n' + '\n')
            t4 = time.time()
            # my_json['prompt'] = prompt
            # my_json['label'] = label
            # file.write(json.dumps(my_json))
            # file.write('\n')
            # encoded_input = self.llmTokenizer(prompt, padding=True, return_tensors='pt')
            # output = self.llmModel.generate(encoded_input["input_ids"].cuda(0), max_new_tokens=10)
            response, history= self.llmModel.chat(self.llmTokenizer, prompt1, history=[])
            t5 = time.time()
            # print("LLM spend time: {}".format(t5-t4))
            llm_time += t5-t4
            res = ''
            if response.find("BENIGN") != -1 or response.find("Benign") != -1:
                res = "Benign"
            elif response.find("Unknown") != -1 or response.find("UNKNOWN") != -1:
                res = str(maxlabel)
            else:
                res = "Attack"
            print(label, response, maxlabel)
            string = '\"real\":\"'+ str(label) + '\",\"predict\":\"' + str(res) + '\",\"response\":\"' + str(response) + '\",\"maxlabel\":\"' + str(maxlabel) + '\"'
            log2.write("{" + string + "}\n")
            if response == label:
                right += 1
            else:
                wrong += 1
            t6 = time.time()
            # print("One sample spend time: {}".format(t6-t1))
            cost_time += t6-t1
            self.vectorDatabase[item] = base
        # file.close()    
        acc = right / cnt
        print("Accuracy: {}%".format(acc * 100))
        log1.write("Similarity spend all time: " + str(similarity_time) + '\n')
        log1.write("LLM spend all time: " + str(llm_time) + '\n')
        log1.write("Spend all time: " + str(cost_time) + '\n')
        log1.write("Accuracy: {}%".format(acc * 100))
        log1.close()
        log2.close()
        
        





            
if __name__ == '__main__':

    prompt = "### Characterization " + '\n' \
            + "You're an intrusion detection system that specializes in detecting abnormal network data." + '\n' \
            + "### Data form" + '\n' \
            + "Network data has been converted into hexadecimal data, each network data has 82 features, each feature is separated by \"0x\"." + '\n' \
            + "### Task" + '\n' \
            + "Now that you have real network traffic data, you need to determine its category. You need to follow a template for the output." + '\n' \
            + "-Input format: a hexadecimal network data" + '\n' \
            + "-Output format: its corresponding class label" + '\n' \
            + "### Steps" + '\n' \
            + "1. Parse the input hexadecimal data to extract the network data." + '\n' \
            + "2. Extract the necessary features from the parsed data." + '\n' \
            + "3. Classify the extracted features into their corresponding categories." + '\n' \
            + "4. Output the class label for each network data." + '\n' \
            + "### Sample" + '\n' \
            + "{a}" + '\n' \
            + "### Notes" + '\n' \
            + "- Follow the steps strictly." + '\n' \
            + "- Strictly follow the output format and output the data's class label directly, without outputting redundant content." + '\n' \
            + "Now there is some web data for you to test:" + '\n' + "{b}"

    # model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision = "v1.0.0")
    # tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # llmModel = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    # llmModel = llmModel.eval()

    path = './dataset'
    map1 = {}



    print(map1)

    print("OK")



