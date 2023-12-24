import pandas as pd
import numpy as np
import os


def process():
    filename=os.listdir(os.getcwd())
    for f in filename:
        if os.path.isdir(os.getcwd()+"//"+f):
            data_train=pd.read_csv(os.getcwd()+"//"+f+"//train.txt",sep=" ",dtype=int,names=["SessionId","ItemId"])
            data_test=pd.read_csv(os.getcwd()+"//"+f+"//test.txt",sep=" ",dtype=int,names=["SessionId","ItemId"])
            data_train["TimeStamp"]=0
            data_test["TimrStamp"]=0
            data_train.to_csv(os.getcwd()+"//"+f+"//train.tsv",sep="\t",index=False)
            data_test.to_csv(os.getcwd()+"//"+f+"//test.tsv",sep="\t",index=False)





if __name__ == '__main__':
    process()