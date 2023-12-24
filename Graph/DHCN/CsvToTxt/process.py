import pandas as pd
import numpy as np

def processData(file):
    data=pd.read_csv(file)
    s=""
    print(data["user_id"].unique())
    for i in data["user_id"].unique():
        item=data.loc[data["user_id"]==i,"item_id"]
        s+=str(i)
        for j in item:
            s+=" "
            s+=str(j)
        s+="\n"
    with open(file[:-4]+".txt","w+") as f:
        f.write(s)








import pickle
def txt_to_csv(path):
    fname=os.listdir(path)
    for fn in fname[::-1]:
        fn="\\"+fn+"\\"
        for fi in os.listdir(path+fn):
            d=open(path+fn+"\\"+fi,"rb+")
            #d=d.readlines()
            #ft=f.readlines()
            dt=pickle.load(d)
            print(1,fi,dt,1)
            #with open("DHCN"+fn+fi[:-3]+"txt") as f:
            #    f.write()
            dt_f=pd.DataFrame(dt)
            dt_f.to_csv("DHCN"+fn+fi[:-3]+"csv")

def processData1(file):
    data=pd.read_csv(file)
    s=""
    for i in range(1,data.shape[1]+1):
        s+=str(i)
        item=data.iloc[0,i-1]
        ground_truth=data.iloc[1,i-1]
        print(item,ground_truth)
        break
    #with open(file[:-4]+".txt","w+") as f:
    #    f.write(s)

def process(path):
    filenames=os.listdir(path)
    for f in filenames:
        print(f)
        if f in ["train.csv","test.csv"]:
            print(path+"\\"+f)
            processData1(path+"\\"+f)


def tackle_data(path):
    data_train, groundtruth_train = pickle.load(open(path+"\\train.txt", 'rb'))
    data_test, groundtruth_test = pickle.load(open(path+"\\train.txt", 'rb'))
    length_train = len(data_train)
    length_test=len(data_test)
    print(length_train,length_test)
    for mode in ["crop","pad"]:
        temp_train_data=[]
        temp_test_data=[]
        temp_train_g=[]
        temp_test_g=[]
        with open(path+"//total_"+mode+".txt", "w") as f:
            if mode=="crop": #剪切
                for it in range(length_train):
                    if len(data_train[it])<4:
                        length_train-=1
                        continue
                    temp_train_data.append(data_train[it])
                    temp_train_g.append(groundtruth_train[it])

                for jt in range(length_test):
                    if len(data_test[jt])<4:
                        length_test-=1
                        continue

                    temp_test_data.append(data_test[jt])
                    temp_test_g.append(groundtruth_test[jt])


                for i in range(len(temp_train_data)):
                    f.write(str(i + 1) + " " + " ".join(map(str, temp_train_data[i])) + " " + str(temp_train_g[i]) + "\n")


                for j in range(len(temp_test_data)):
                    f.write(str(j +len(temp_train_data)) + " " + " ".join(map(str, temp_test_data[j])) + " " + str(temp_test_g[j]) + "\n")
            else: #填充
                for it in range(len(data_train)):
                    while len(data_train[it])<3:
                        data_train[it].insert(0,0)

                for jt in range(len(data_test)):
                    while len(data_test[jt])<3:
                        data_test[jt].insert(0,0)

                for i in range(len(data_train)):
                    f.write(str(i + 1) + " " + " ".join(map(str, data_train[i])) + " " + str(groundtruth_train[i]) + "\n")


                for j in range(len(data_test)):
                    f.write(str(j +len(data_train)) + " " + " ".join(map(str, data_test[j])) + " " + str(groundtruth_test[j]) + "\n")





def concat(path):

    with open(path+"//train_temp.txt","r+") as f:
        data_train=f.readlines()

    data_train_crop=[i for i in data_train if len(" ".split(i))>4]
    data_train_pad=[]
    for i in data_train:
        if len(" ".split(i))<4:
            i=i+" 0"*(4-len(" ".split(i)))
        data_train_pad.append(i)


    with open(path + "//test_temp.txt", "r+") as f:
        data_test = f.readlines()
    data_test_crop = [i for i in data_test if len(" ".split(i)) > 4]
    data_train_pad = []
    for i in data_train:
        if len(" ".split(i)) < 4:
            i = i + " 0" * (4 - len(" ".split(i)))
        data_train_pad.append(i)






if __name__ == '__main__':
    import os
    choose=int(input("input 1:raw data,input 2 csv_data :"))
    if choose==1:
    # txt_to_csv
        path=os.getcwd()+"\..\DHCN\datasets_DHCN"
        txt_to_csv(path)
    else:
        path=os.getcwd()+"\\DHCN"
        filenames = os.listdir(path)
        for f in filenames:
            t_path=path+"\\"+f
            #print(t_path)
            tackle_data(t_path)














