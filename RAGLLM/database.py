# -*- coding: utf-8 -*-

import csv
import json 
import pandas as pd
import numpy as np
import random
import os
from transformers import AutoTokenizer, AutoModel
# from modelscope import AutoTokenizer, AutoModel
import torch
import encode



def searchfile(Path, File):
    li = []
    files = os.listdir(Path)
    for f in files:
        if os.path.isdir(Path+os.sep+f):
            searchfile(Path+os.sep+f, File)
        elif f.split('.')[-1] == File:
            li.append(f)
    return li

class DataBase:

    def __init__(self, vectorDatabase, testSet, embeddingModel, embeddingTokenizer = None):
        self.vectorDatabase = vectorDatabase
        self.embeddingModel = embeddingModel
        self.embeddingTokenizer = embeddingTokenizer
        self.testSet = testSet
        
    def createDatabaseAndTestset_encode(self, folders1):
        
        f1 = searchfile(folders1, 'csv')
        for c in f1:
            file_path = folders1+os.sep+c
            print(c)
            df = pd.read_csv(file_path, header=0, low_memory=False)
            col_list = next(csv.reader(open(file_path), delimiter=','))
            if col_list[0] == '':
                df = df.drop(df.columns[0],axis=1)
                col_list.pop(0)
            while col_list[-1] != 'Label' and col_list[-1] != 'label':
                df = df.drop(df.columns[-1],axis=1)
                col_list.pop(-1)
            X = df.to_numpy()
            # print(len(col_list))
            df_c = X[:,-1].tolist()  
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            X = df.to_numpy()
            df_b = X[ : , : -1].astype(np.float32)
            s = df_b.shape
            zero = torch.zeros(s[0],128-s[1])
            data = torch.from_numpy(df_b)
            data_a = torch.cat([data, zero],dim=1)
            print(len(data_a))
            le = len(df_b)
            res1 = random.sample(range(0,le), min(100, le)) # 构建数据库
            res2 = random.sample(range(0,le), min(2000, le)) # 测试集查询
            for k in res1:
                s2 = ''
                # if df_c[k] != 'BENIGN':
                #     label = 'Attack'
                # else:
                #     label = df_c[k]
                label = df_c[k]
                for j in range(len(df_b[k])): 
                    s2 += str(col_list[j]) + ":" + str(df_b[k][j])+','
                    # s += str(hex(round(df_b[k][j])))
                # print(v)
                a = self.embeddingModel(data_a[k])[0]
                a = a.detach().numpy()
                v = np.append(a,label)
                self.vectorDatabase[s2] = v
            for k in res2:
                s2 = ''
                # if df_c[k] != 'BENIGN':
                #     label = 'Attack'
                # else:
                #     label = df_c[k]
                label = df_c[k]
                for j in range(len(df_b[k])): 
                    s2 += str(col_list[j]) + ":" + str(df_b[k][j])+','
                    # s += str(hex(round(df_b[k][j])))
                a = self.embeddingModel(data_a[k])[0]
                a = a.detach().numpy()
                v = np.append(a,label)
                self.testSet[s2] = v
            del df
            del df_b
            del df_c
            



            
if __name__ == '__main__':

    path = './dataset'
    vectorDatabase = {}

    tokenizer = AutoTokenizer.from_pretrained('./model/bge-large-en-v1.5')
    model = AutoModel.from_pretrained('./model/bge-large-en-v1.5')
    model.eval()

    dataBase = DataBase(vectorDatabase, model, tokenizer)
    dataBase.createDatabase(path)
    # for item in dataBase.vectorDatabase:
    #     print(dataBase,vectorDatabase[item][:-1].astype(np.float64))

    print("OK")



