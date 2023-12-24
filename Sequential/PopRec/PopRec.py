import os

from dataProcessing import *
from utils import *


def PopRec(original_data,actual_data,data_dic,topk):
    data_dic=sorted(list(data_dic.keys()),key=lambda x:data_dic[x],reverse=True)
    pred=[[] for _ in range(len(actual_data))]
    for i in range(len(actual_data)):
        for j in data_dic:
            if len(pred[i])<topk and j not in original_data[i]: # 保证了推荐的item不在user的交互历史中
                    pred[i].append(j)
    return pred,actual_data



def metric_all(tag):
    filename=os.listdir(os.getcwd()+"/data/")
    with open("result.txt","w+") as ff:
        for f in filename:
            if f[-4:] !=".txt":
                continue
            print(f)
            # 做next_item predict
            original_data, actual_data, data_dic,_=dataProcess("/data/"+f,topk=1)
            ff.write(f+"\n")
            for t in tag:
                pred, actual= PopRec(original_data, actual_data, data_dic,t)
                rc_t=round(recall_at_k(actual,pred,t),4)
                ndcg_t=round(ndcg_k(actual,pred,t),4)
                print('HIT@{0} : {1} , NDCG@{0} : {2}'.format(t,rc_t,ndcg_t))
                ff.write('HIT@{0} : {1} , NDCG@{0} : {2}'.format(t,rc_t,ndcg_t))
                ff.write("\n")

if __name__ == '__main__':
    tag=[1,5,10,15,20,25,30]
    metric_all(tag)









