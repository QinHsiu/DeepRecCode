import random

from dataProcessing import *
from utils import *
import numpy as np
import math
# 原理参考：https://www.jianshu.com/p/f306a37a7374


def cal_sim(i_a,i_b,a,b,data_dic,item_matrix):
    """
    :param a:item a id
    :param b:item b id
    :param data_dic: item count
    :param item_matrix:item matrix
    :return:
    """
    return item_matrix[i_a,i_b]/math.sqrt(data_dic[a]*data_dic[b]) # 打分





def itemCF(original_data,actual_data,data_dic,topk=1):
    # 创建一个n*n的矩阵，n表示item的数目
    item_n=len(data_dic.keys()) # 不同item数目
    item_matrix=np.zeros((item_n,item_n)) # 创建矩阵,其中matrix[a,b]表示item a与item b共现次数
    # 创建一个列表index与字典中item的映射关系
    item_ids=list(data_dic.keys())
    # 使用集合记录所有交互过的item
    item_set=set(item_ids)
    # 存储每个用户没有交互过的item集合
    non_item_set=[item_set for _ in range(len(actual_data))]
    # 初始化item_matrix矩阵
    for i_ in range(len(original_data)): # 对于每一个user
        for j_ in range(len(original_data[i_])): # 对于每一个交互的item
            for i_j_ in range(j_,len(original_data[i_])):
                temp_i=item_ids.index(original_data[i_][j_])
                temp_j=item_ids.index(original_data[i_][i_j_])
                if temp_j!=temp_i:
                    item_matrix[temp_i,temp_j]+=1
        non_item_set[i_]=non_item_set[i_]-set(original_data[i_])
    pred=[[] for _ in range(len(actual_data))]
    for i in range(len(actual_data)): # 需要推荐的用户数目
        temp_dict={}
        for j_ in non_item_set[i]:
            temp_i_j=0
            for i_ in original_data[i]:
                t_i=item_ids.index(i_)
                t_j=item_ids.index(j_)
                if item_matrix[t_j,t_i]==0.0:
                    continue
                temp_i_j+=cal_sim(t_i,t_j,i_,j_,data_dic,item_matrix)

            temp_dict[j_]=temp_i_j
        pred[i].extend(sorted(temp_dict.keys(),key=lambda x:temp_dict[x],reverse=True)[:topk])

    return pred,actual_data


def item_CF(original_data,actual_data,data_dic,us_it,mode,topk=1):
    """
    :param original_data: [[1,2,3]···]
    :param actual_data: [[1],[2]···]
    :param data_dic: {1:2,3:4···}
    :param topk: 1
    :return:
    """
    pred_data=[[] for _ in range(len(original_data))]
    for l in range(len(pred_data)):
        pred_data[l].append(original_data[l][-1])
    C={}
    N={}

    for idx,(u,items) in enumerate(us_it.items()):
        if mode=="ItemCF":
            for i in items.keys(): # 对于每一个user
                N.setdefault(i,0)
                N[i]+=1
                for j in items.keys():
                    if i==j:
                        continue
                    C.setdefault(i,{})
                    C[i].setdefault(j,0)
                    C[i][j]+=1
        else:
            for i in items.keys(): # 对于每一个user
                N.setdefault(i,0)
                N[i]+=1
                for j in items.keys():
                    if i==j:
                        continue
                    C.setdefault(i,{})
                    C[i].setdefault(j,0)
                    C[i][j]+=1/math.log(1+len(items)*1.0)

        # print(C)
    itemSimBest={}
    for idx,(cur_item,related_items) in enumerate(C.items()):
        itemSimBest.setdefault(cur_item,{})
        for related_item,score in related_items.items():
            itemSimBest[cur_item].setdefault(related_item,0)
            itemSimBest[cur_item][related_item]+=score/math.sqrt(N[cur_item] * N[related_item])
    itemSimbest_f={}
    for it in itemSimBest.keys():
        itemSimbest_f[it]=sorted(itemSimBest[it].items(),key=lambda x:x[1],reverse=True)


    # print(itemSimbest_f)
    pred=[[] for _ in range(len(original_data))]

    for u_i in range(len(pred)):
        temp=[]
        if pred_data[u_i][0] in itemSimbest_f:
            for i_i in itemSimbest_f[pred_data[u_i][0]]:
                if i_i[0] not in original_data[u_i] and len(temp)<topk:
                    temp.append(i_i[0])
                else:
                    continue
            pred[u_i].extend(temp)
            while len(pred[u_i])<topk:
                t_=random.randint(1,len(actual_data))
                if t_ not in original_data[u_i]:
                    pred[u_i].append(t_)
                else:
                    continue
        else:
            i=0
            while i<topk:
                t_=random.randint(1,len(actual_data))
                if t_ not in original_data[u_i]:
                    pred[u_i].append(t_)
                    i+=1
                else:
                    continue
    return pred,actual_data



def test():
    o=[[1,2,3,4],[2,3,4,5,1],[1,2,6,4]]
    a=[[5],[6],[3]]
    dic={1:3,2:3,3:2,4:3,5:1,6:1}
    us_it={1:{1:1,2:1,3:1,4:1},2:{2:1,3:1,4:1,5:1,1:1},3:{1:1,2:1,6:1,4:1}}
    print(item_CF(o,a,dic,us_it,"ItemCF_IUF",8))



def metric_all(tag):
    filename=os.listdir(os.getcwd()+"/data/")
    with open("ItemCF_result.txt","w+") as ff:
        for f in filename:
            if f[-4:] !=".txt" or f[:-4]!='ml-1m':
                continue
            print(f)
            # 做next_item predict
            original_data, actual_data, data_dic, us_it = dataProcess("/data/" + f, topk=1)
            ff.write(f+"\n")
            pred, actual = item_CF(original_data, actual_data, data_dic, us_it, "ItemCF_IUF", 100)
            for t in tag:
                rc_t=round(recall_at_k(actual,pred,t),4)
                ndcg_t=round(ndcg_k(actual,pred,t),4)
                print('HIT@{0} : {1} , NDCG@{0} : {2}'.format(t,rc_t,ndcg_t))
                ff.write('HIT@{0} : {1} , NDCG@{0} : {2}'.format(t,rc_t,ndcg_t))
                ff.write("\n")





if __name__ == '__main__':
    tag = [1, 5, 10, 15, 20, 25, 30,50]
    metric_all(tag)
    # test()


