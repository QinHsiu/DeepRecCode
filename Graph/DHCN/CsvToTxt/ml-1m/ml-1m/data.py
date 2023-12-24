import os
import gzip
import json
import math
import random
import pickle
import pprint
import argparse

import numpy as np
import pandas as pd


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError

def processData(file,data):
    #data=pd.read_csv(file)
    s=""
    print(data["user"].unique())
    for i in data["user"].unique():
        item=data.loc[data["user"]==i,"item"]
        s+=str(i)
        for j in item:
            s+=" "
            s+=str(j)
        s+="\n"

    with open(file[:-4]+".txt","w+") as f:
        f.write(s)

class Basline(DatasetLoader):
    def __init__(self,data_dir,f):
        self.fpath=os.path.join(data_dir,f)
    def load(self):
        with open(self.fpath,"r+") as f:
            data=f.readlines()
        train_data=[]
        test_data=[]
        actual_data=[]
        item=[]
        user=[]
        i=0
        for d in data:
            temp = list(map(int, d.split(" ")))
            for t_ in temp[1:]:
                user.append(i)
                item.append(t_)
            i+=1
            train_data.append(temp[1:-1])
            test_data.append(temp[1:-1])
            actual_data.append([temp[-1]])
        df=pd.DataFrame()
        df["user"]=user
        df["item"]=item
        return train_data,test_data,df,actual_data

class MovieLens1M(DatasetLoader):
    def __init__(self, data_dir):
        self.fpath = os.path.join(data_dir, 'ratings.dat')

    def load(self):
        # Load data
        df = pd.read_csv(self.fpath,
                         sep='::',
                         engine='python',
                         names=['user', 'item', 'rate', 'time'])
        # TODO: Remove negative rating?
        # df = df[df['rate'] >= 3]
        return df


if __name__ == '__main__':
    train=MovieLens1M(os.getcwd()).load()
    #print(len(movie["user"].unique().tolist()))
    #print(movie)
    #processData("Movie-1m",movie)
    #data=pd.read_csv(os.getcwd()+"/Movi.txt")
    # with open(os.getcwd()+"/Movi.txt","r+") as f:
    #     p=f.readlines()
    # user_list=[]
    # for d in p:
    #     user_list.append(list(map(int,d.split(" ")[1:])))












