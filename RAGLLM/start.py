from transformers import AutoTokenizer, AutoModel
# from modelscope import AutoTokenizer, AutoModel
import csv
import json 
import pandas as pd
import numpy as np
import random
import os
import database
import query
import encode
import torch


if __name__ == '__main__':
    vectorDatabase = {}
    testSet = {}
    
    path = '/public/home/ai_user_3/dataset/2018_dataset'
    
    logPath1 = './log-18-cnn17-prompt-3-100-all-2.txt'
    
    logPath2 = './log-18-cnn17-result-3-100-all-2.txt'
    
    
    prompt =  "### Role " + '\n' \
            + "You're an intrusion detection system that specializes in predicting the class label of network data." + '\n' + '\n'\
            + "### Task" + '\n' \
            + "Now that you have real network traffic data, you need to determine its class label. You need to output the class label of it." + '\n' \
            + "-Input format: a real network data" + '\n' \
            + "-Output format: The class label of the input." + '\n' \
            + "### Sample" + '\n' \
            + "{a}" + '\n' \
            + "### Notes" + '\n' \
            + "- Please predict the class label of network data according to the given samples." + '\n' \
            + "- If you can't predict, just output Attack." + '\n' + '\n'\
            + "### There is a network data, please just output the class label of it, don't output analysis and explanation: "  + '\n'\
            + "{b}" + "\n" 
            # + "Output: "
    # "### Sample" + '\n' \
    #         + "{a}" + '\n' \
    #         + "Determine the labels for the following network types based on the sample above, noting that just the output the label：" + '\n' \
    #         + "Input: " + "\n" + "{b}" + "\n" \
    #         + "Output: "
            # + "### The input network data is:" + '\n' \
            # + "### Data form" + '\n' \
            # + "Network data has 82 features, ." + '\n' \
        #     + "### Steps" + '\n' \
        #     + "1. Parse the input hexadecimal data to extract the network data." + '\n' \
        #     + "2. Extract the necessary features from the parsed data." + '\n' \
        #     + "3. Classify the extracted features into their corresponding categories." + '\n' \
        #     + "4. Output the class label for each network data." + '\n' \
    
#     embeddingModel_dir = '/public/home/ai_user_3/model/bge-large-en-v1.5'
#     embeddingTokenizer = AutoTokenizer.from_pretrained(embeddingModel_dir)
#     embeddingModel = AutoModel.from_pretrained(embeddingModel_dir)
#     embeddingModel = embeddingModel.eval()
    
#     dataBase = database.DataBase(vectorDatabase, testSet, embeddingModel, embeddingTokenizer)
#     dataBase.createDatabaseAndTestset_embedding(path)

    embeddingModel = encode.MLP(128, 15)
    if os.path.exists("/public/home/ai_user_3/model/Encode/all-cnn-encode.pkl"):
        embeddingModel.load_state_dict(torch.load("/public/home/ai_user_3/model/Encode/all-cnn-encode.pkl"))
        print("load model success")
    embeddingModel = embeddingModel.eval()

    dataBase = database.DataBase(vectorDatabase, testSet, embeddingModel)
    dataBase.createDatabaseAndTestset_encode(path)
    
    
    llmModel_dir = "/public/home/ai_user_3/model/chatglm3-6b"
    llmTokenizer = AutoTokenizer.from_pretrained(llmModel_dir, trust_remote_code=True)
    llmModel = AutoModel.from_pretrained(llmModel_dir, trust_remote_code=True).cuda(0)
    llmModel = llmModel.eval()




    # print(vectorDatabase)

    queryModel = query.query(vectorDatabase, testSet, llmModel, llmTokenizer)

    queryModel.getResult(prompt, logPath1, logPath2)

