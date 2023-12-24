import numpy as np
import pandas as pd
import os

"""根据流行度对用户进行推荐，不需要训练集、验证集,区分真实数据与原始数据"""
"""根据协同过滤对用户进行推荐，不需要训练集、验证集,区分真实数据与原始数据"""
def dataProcess(filename,topk):
    """
    :param:
    filename:需要处理的数据文件名
    topk:最好的k个推荐
    :return:
     original_data: 原始数据（去掉最后一个item的数据）
     data_dic:原始数据形成的数据字典
     actual_data:每个user的最后一个item数据
    """
    original_data=[]
    train_data=[]
    data_dic={}
    actual_data=[]
    us_it={}
    with open(os.getcwd()+filename) as f:
        d_f=f.readlines()
    for d in d_f:
        temp=list(map(int,d.split(" ")))
        us_it.setdefault(temp[0],{}) # 为每一个user 分配一个字典
        for it_ in temp[1:-1*topk]:
            us_it[temp[0]][it_]=1
        train_data.append(temp[1:-1*topk])
        original_data.extend(temp[1:-1*topk])
        actual_data.append(temp[-topk:])

    for i_ in original_data:
        data_dic[i_]=data_dic.get(i_,0)+1

    return train_data,actual_data,data_dic,us_it



if __name__ == '__main__':
    filename="/data/Sports_and_Outdoors.txt"
    train,test,_,us_it=dataProcess(filename,1)
    print(us_it)







