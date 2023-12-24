from dataProcessing import *
from utils import *
import numpy as np
import math
# 原理参考：https://www.jianshu.com/p/7c5d9c008be9

def cal_sim(a,b,original_data):
    return len(set(original_data[a]).intersection(set(original_data[b])))/math.sqrt(len(original_data[a])*len(original_data[b]))

def UserCF(original_data, actual_data,t):
    # t表示top_k个相似的user
    user_matrix=np.zeros((len(original_data),len(original_data)))
    # 初始化矩阵
    for u_ in range(len(original_data)):
        for u_i in range(u_+1,len(original_data)):
            user_matrix[u_,u_i]=cal_sim(u_,u_i,original_data)
            user_matrix[u_i, u_] =user_matrix[u_,u_i]
    print(user_matrix)
    pred=[[] for _ in range(len(original_data))]
    for u_ in range(len(original_data)):
        temp=user_matrix[u_]
        temp_dic=dict(zip([i for i in range(len(temp))],temp))
        temp_user=sorted(temp_dic.keys(),key=lambda x:temp_dic[x],reverse=True)[:t] # 对于相似的user进行排序，取topk个
        item_set=set()
        item_count={}
        for u_t in temp_user:
            item_set.update(set(original_data[u_t]).difference(set(original_data[u_])))

        item_set=list(item_set)
        for u_i in temp_user:
            for i_i in item_set:
                if i_i in original_data[u_i]:
                    if i_i not in item_count:
                        item_count[i_i]=1
                    else:
                        item_count[i_i]+=1

        item_set=sorted(item_set,key=lambda x:item_count[x],reverse=True)
        pred[u_].extend(item_set[:1])
    return pred,actual_data

def User_CF(original_data, actual_data,us_it,t):
    U_U={}
    U_I={}
    for u_,itemsets in enumerate(original_data):
        U_U.setdefault(u_,{})
        for u_n,itemsets in enumerate(original_data):
            if u_==u_n:
                U_U[u_][u_n]=-1
                continue
            U_U[u_][u_n]=cal_sim(u_,u_n,original_data)
        temp_userset=sorted(U_U[u_].keys(),key=lambda x:U_U[u_][x],reverse=True)
        temp_itemset=[]
        # print(temp_userset)
        for u_u in temp_userset: # 4 2 3 -> 3 1 2
            # print(u_u)
            if len(set(temp_itemset))<t:
                temp_itemset.extend(list(set(original_data[u_u])-set(original_data[u_])))
            else:
                break
        temp_dic={}
        for i_t_t in temp_itemset:
            if i_t_t not in temp_dic:
                temp_dic[i_t_t]=1
            else:
                temp_dic[i_t_t]+=1

        U_I.setdefault(u_,[])
        temp_itemset=sorted(list(set(temp_itemset)),key=lambda x:temp_dic[x],reverse=True)[:t]
        U_I[u_].extend(temp_itemset)

    return list(U_I.values()),actual_data



def test():
    original_data=[[1,2,3],[1,4,5],[1,5,6],[2,4]]
    actual_data=[[5],[2],[3],[6]]
    # pred,actual_data=UserCF(original_data,actual_data,3)
    us_it=None
    pred,actual_data=User_CF(original_data, actual_data, us_it, 5)
    print(pred,actual_data)






def metric_all(tag):
    filename=os.listdir(os.getcwd()+"/data/")
    with open("UserCF_result.txt","w+") as ff:
        for f in filename:
            if f[-4:] !=".txt" or f[:5]!='ml-1m':
                continue
            print(f)
            # 做next_item predict
            original_data, actual_data, data_dic,us_it=dataProcess("/data/"+f,topk=1)
            ff.write(f+"\n")
            # pred, actual = User_CF(original_data[:10000], actual_data[:10000], us_it, 100)
            pred, actual = User_CF(original_data, actual_data, us_it, 100)
            for t in tag:
                pred_t=[pred[i][:t] for i in range(len(pred))]
                rc_t=round(recall_at_k(actual,pred_t,t),4)
                ndcg_t=round(ndcg_k(actual,pred_t,t),4)
                print('HIT@{0} : {1} , NDCG@{0} : {2}'.format(t,rc_t,ndcg_t))
                ff.write('HIT@{0} : {1} , NDCG@{0} : {2}'.format(t,rc_t,ndcg_t))
                ff.write("\n")

if __name__ == '__main__':
    tag = [1, 5, 10, 15, 20, 25, 30,50]
    metric_all(tag)
    # test()