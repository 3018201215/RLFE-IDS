import csv
import pandas as pd
import numpy as np
import os
import time
from PIL import Image
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim import SGD
from tqdm import tqdm
import time
import math
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
        )
        self.net3 = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )          
        self.net4 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(1, stride=2),
        )

        self.fullconnection = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
        )

    
    def forward(self, x):
        x = torch.reshape(x,(-1, 1, self.in_dim))
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = torch.reshape(x,(-1, int(self.in_dim*32)))
        x = self.fullconnection(x)
        return x
# class MLP(nn.Module):
#     def __init__(self, input_feature, num_class) -> None:
#         super(MLP, self).__init__()
#         self.in_dim = input_feature
#         self.num_class = num_class


#         self.net1 = nn.Sequential(
#             nn.Conv1d(1, 32, 1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#         )
#         self.net2 = nn.Sequential(
#             nn.Conv1d(32, 64, 1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.AvgPool1d(1, stride=2),
#         )
        

#         self.fullconnection = nn.Sequential(
#             nn.Linear(int(self.in_dim*32), 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 1024),
#             nn.ReLU(),
#             # nn.Linear(512, 1024),
#             # nn.ReLU(),
#         )

#         # self.softmax = nn.Softmax(1)
    
#     def forward(self, x):
#         x = torch.reshape(x,(-1, 1, self.in_dim))
#         x = self.net1(x)
#         x = self.net2(x)
#         x = torch.reshape(x,(-1, int(self.in_dim*32)))
#         x = self.fullconnection(x)
#         return x

