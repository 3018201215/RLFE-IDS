import csv
import pandas as pd
import numpy as np
import os
import time
from PIL import Image
import scipy
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from torch.optim import SGD
from tqdm import tqdm
import time
import math
import random
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_feature, num_class) -> None:
        super(MLP, self).__init__()
        self.in_dim = input_feature
        self.num_class = num_class

        self.net1 = nn.Sequential(
            nn.Conv1d(1, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.AvgPool1d(1, stride=2),
        )
        self.net3 = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
# #           
        self.net4 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(1, stride=2),
        )
# #             nn.Conv1d(in_channels=256, out_channels=256, kernel_size = 2)
        self.fullconnection = nn.Sequential(
            nn.Linear(int(self.in_dim*32), 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),

        )

    
    def forward(self, x, support_set):
        meta_dict = torch.zeros(size = [support_set.shape[0], 1024])
        for i in range(support_set.shape[0]):
            d = support_set[i]
            d = torch.reshape(d,(-1, 1, self.in_dim))
            d = self.net1(d)
            d = self.net2(d)
            d = self.net3(d)
            d = self.net4(d)
            d = torch.reshape(d,(-1, int(self.in_dim*32)))
            d = self.fullconnection(d)
            meta_dict[i] = torch.mean(d, dim = 0)
        x = torch.reshape(x,(-1, 1, self.in_dim))
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = torch.reshape(x,(-1, int(self.in_dim*32)))
        x = self.fullconnection(x)
        N_query = x.shape[0]
        result = torch.zeros(size=[N_query, support_set.shape[0]])
        pdist = nn.PairwiseDistance(p=2)
        meta_dict = meta_dict.to(device)
        for i in range(0, N_query):
            temp_value = x[i].repeat(support_set.shape[0], 1)
            dist_value = pdist(meta_dict, temp_value)
            result[i] = dist_value
        result = F.log_softmax(result, dim = 1)
        return result
    
def searchfile(Path, File):
    list = []
    files = os.listdir(Path)
    for f in files:
        if os.path.isdir(Path+os.sep+f):
            #递归查找
            searchfile(Path+os.sep+f, File)
        elif f.split('.')[-1] == File:
            list.append(Path+os.sep+f)
    return list

def ProcessCsv(file_path, device):
#     print('1 ok')
    postfix = 'csv'
    list1 = searchfile(file_path, postfix)
    DF = []
    for f in list1:
#         print('2 ok')
        file = os.path.basename(f)
        # if file != "WebDDos.csv":
        #     continue
        print('开始处理文件 {}'.format(file))
        df1 = pd.read_csv(f, low_memory=False,header=0)
        # print(df1.shape)
        if df1.shape[0] > 100 :
            df1 = df1.sample(frac = 0.1, axis = 0)
        df1 = df1.drop(df1.columns[0],axis=1)
        # print(df1.shape)
        DF.append(df1)
        # print(len(DF))
        print('{} 文件处理完成，'.format(file))
#         print('3 ok')
        # print(file_path+os.sep+f)
#     file_path = r'E:\DataBase\Pycharm\HJW_classification_traffic\DDoS_classification\Data\SAT-03-11-2018_03.pcap'   # 测试用
        
       
#         break
#         if num >= 5:
#             break
    df = pd.concat(DF)
    # print(df.columns)
    # print(df.shape)
#     df2.to_csv('./2018.csv')
    
#     df = pd.read_csv('./2018.csv')
    
    #Read dataset
#     df = pd.read_csv(file_path, low_memory=False, header=None)
    X = df.to_numpy()
    df_c = X[:,-1]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    X = df.to_numpy()
    df_b = X[ : , : -1] 
#     de = []
#     for i in range(len(X)):
#         if X[i][0] == 'Dst Port':
# #             print(X[i])
#             de.append(i)
#     df_a = np.delete(X, de, 0)
    labelencoder = LabelEncoder()
    # for i in range(len(df_c)):
    #     df_c[i] = df_c[i].replace("DrDoS_", "")
    #     df_c[i] = df_c[i].replace("-", "")
        # df_c[i] = df_c[i].replace("l", "L")
    df_c = labelencoder.fit_transform(df_c)
    data=[]
    df_b = df_b.astype(np.float32)
    y = df_c.astype(np.int32)
    s = df_b.shape
    zero = torch.zeros(s[0],128-s[1])
    data = torch.from_numpy(df_b)
#     min_val = torch.min(data, dim=0).values
#     max_val = torch.max(data, dim=0).values
#     data_t = (data-min_val)/(max_val-min_val)
#     print(data_t)
    data_a = torch.cat([data, zero],dim=1)
    print(data_a.shape)
#     data_b = torch.reshape(data_a, (-1,16,16))
#     print(data_b.shape)
#     return X
#     y = label.reshape(-1,1)
    y = np.ravel(y)
    y = torch.from_numpy(y)
#     data_cpu = data.cpu()
    support_set = torch.zeros(size = [len(list1), 10, 128])
    for i in range(support_set.shape[0]):
        l = []
        for j in range(y.shape[0]):
            if y[j] == i:
                l.append(data_a[i])
        k = random.sample(range(0,len(l)), 10)
        s1 = []
        for m in k:
            s1.append(l[m])
        samples = torch.stack(s1)
        support_set[i] = samples
    X_train, X_test, y_train, y_test = train_test_split(data_a, y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)
    print(X_train.shape)
    # print(pd.Series(y_train).value_counts())
#     print(labelencoder.inverse_transform([0,1,2,3,4,5]))
    return X_train, X_test, y_train, y_test, support_set
                
class MyDataset():  #加载数据与整理数据
    def __init__(self,data, label):
#         normalize = transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        self.data = data
        self.label = label
#         print(len(data))
#         print(len(label))
#         self.transfrom = transforms.Compose([transforms.Resize(size=(1,78)), transforms.ToTensor()])    
        
    def __getitem__(self,index):
        return self.data[index],self.label[index]
        
    def __len__(self):
        return len(self.data)

    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
        
def __train(dl, model, optim, BS, i, f, support_set):
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0
    it = 0
    acc_sum = 0
    for x, y in tqdm(dl, desc='training'):
        x = x.to(device)
        y = y.to(device)
        _y = model(x, support_set).to(device)
        _, ymax = torch.max(_y, dim=1)
        acc_sum += (ymax == y).sum()
        loss = loss_fn(_y, y.long())
        loss_sum += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        it += 1
    print("Training{}: loss:{}, accurcy rate:{:.2f}".format(i, loss_sum, acc_sum / it / BS * 100))
    f.write("Training{}: loss:{}, accurcy rate:{:.2f}\n".format(i, loss_sum, acc_sum / it / BS * 100))

def __test(dl, BS, i, f, support_set):
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0
    it = 0
    acc_sum = 0
    for x, y in tqdm(dl, desc='testing'):
        x = x.to(device)
        y = y.to(device)
        _y = model(x, support_set).to(device)
        _, ymax = torch.max(_y, dim=1)
        acc_sum += (ymax == y).sum()
        loss = loss_fn(_y, y.long())
        loss_sum += loss.item()
        it += 1
    print("Test{}: loss:{}, accurcy rate:{:.2f}".format(i, loss_sum, acc_sum / it / BS * 100))
    f.write("Test{}: loss:{}, accurcy rate:{:.2f}\n".format(i, loss_sum, acc_sum / it / BS * 100))
    return acc_sum / it / BS * 100

if __name__ == '__main__':
    device= torch.device("cuda:1" if torch.cuda.is_available else "cpu")
    file_path = '/public/home/ai_user_3/dataset/2017_dataset'
    f2 = '/public/home/ai_user_3/dataset/2018_dataset'
    # f3 = '/public/home/ai_user_3/dataset/2019_dataset'
    # X_train, X_test, y_train, y_test, support_set= ProcessCsv(file_path, device)
    X_train1, X_test1, y_train1, y_test1, support_set1= ProcessCsv(f2, device)
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    print('1')

    batch_size = 32
    # trainset = MyDataset(X_train, y_train)
    # testset = MyDataset(X_test, y_test)
    
    trainset1 = MyDataset(X_train1, y_train1)
    testset1 = MyDataset(X_test1, y_test1)
      
# #训练集、测试集、验证集划分        
#     train_loader=DataLoader(trainset,batch_size=batch_size,shuffle=True,drop_last=True)
#     test_loader=DataLoader(testset,batch_size=batch_size,shuffle=False,drop_last=True)
    
    train_loader1=DataLoader(trainset1,batch_size=batch_size,shuffle=True,drop_last=True)
    test_loader1=DataLoader(testset1,batch_size=batch_size,shuffle=False,drop_last=True)

#损失与优化
     #查看是否有GPU可用 
# print(torch.cuda.is_available())
    # support_set = support_set.to(device)
    support_set1 = support_set1.to(device)
    model = MLP(X_train1.shape[1], 15)
    model.apply(weights_init)
    model = model.to(device)
    last_acc = 0.0
    if os.path.exists("./all-cnn-encode.pkl"):
        model.load_state_dict(torch.load("./all-cnn-encode.pkl"))
        print("load model success")
    train_path = './loss_all_cnn_train_2.txt'
    test_path = './loss_all_cnn_test_2.txt'
    f = open(train_path, "w", encoding='utf-8')
    f2 = open(test_path, "w", encoding='utf-8')
    # for i in range(150):
    #     optim = SGD(model.parameters(), lr=1e-3)
    #     __train(train_loader, model, optim, batch_size, i+1, f, support_set)
    #     if((i+1)%5 == 0):
    #         test_acc = __test(test_loader, batch_size, i+1, f, support_set)   
    #         if(test_acc > last_acc):
    #             torch.save(model.state_dict(),str("./all-cnn-encode.pkl"))
    #             last_acc = test_acc
    for i in range(150):
        optim = SGD(model.parameters(), lr=1e-3)
        __train(train_loader1, model, optim, batch_size, i+1, f, support_set1)
        if((i+1)%5 == 0):
            test_acc = __test(test_loader1, batch_size, i+1, f2, support_set1)   
            if(test_acc > last_acc):
                torch.save(model.state_dict(),str("./all-cnn-encode.pkl"))
                last_acc = test_acc
    f.close()
    f2.close()