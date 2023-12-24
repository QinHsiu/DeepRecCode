from .base import BaseModel
from .bert_modules.bert import BERT

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


#######################################MLPENCODER,projection_head
class MLPEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, hidden_dim=4096):
        super(MLPEncoder, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(num_layers - 2):
                #self.linears.append(nn.Linear(input_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))
        #batch normalisation is probably not required in this implementation
        for layer in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, h):
        for layer in range(self.num_layers - 1):
            h = F.gelu(self.batch_norms[layer](self.linears[layer](h)))
        return F.softmax(self.linears[self.num_layers - 1](h))

class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.args=args
        #fixed_bert
        self.bert_fixed = BERT(args,fixed=True)
        torch.save(self.bert_fixed.state_dict(), "bert_fixed_param")
        #variable_bert
        self.bert_variable = BERT(args)
        #copy parameters
        self.bert_variable.load_state_dict(torch.load("bert_fixed_param"))
        self.out = nn.Linear(self.bert_variable.hidden, args.num_items + 1)
        ##projection_head
        self.projectionhead=MLPEncoder(2,self.bert_variable.hidden,self.bert_variable.hidden)

        #pooling function selection
        if args.poolingfunction=="max":
            self.poolingfunction = nn.MaxPool2d((1, self.args.bert_max_len), stride=(1, 1))
        else:
            self.poolingfunction = nn.AvgPool2d((1, self.args.bert_max_len), stride=(1, 1))
    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, input_data):
        # 通过BERT模型 得到输出
        #训练的时候，要输入负样本，得到h_j,c_j
        if type(input_data)==list:
            x,negs=input_data

            h_j, _ = self.bert_fixed(negs)
            c_j, _ = self.bert_variable(negs)
        #测试/验证的时候负样本不在bert模型里面，只有input_data
        else:
            x=input_data
        x_variable,h_i = self.bert_variable(x)#[128*100*256]
        x_fixed,c_i = self.bert_fixed(x)


        #随机选择一个representation
        random.seed(self.args.model_sample_seed)
        #这里的sampler说不定以后可以升级为一个可学习的？比随机的要好
        h_i=random.choice(h_i)
        h_i=self.poolingfunction(h_i).squeeze()
        # 通过一个线性层得到输出
        return self.out(x_variable)


