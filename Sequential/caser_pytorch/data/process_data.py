import numpy as np
import pandas as pd
import os

def dataProcess():
    file_names=os.listdir(os.getcwd())
    for file in file_names:
        if file[-4:]==".txt":
            with open(file,"r+") as f:
                data=f.readlines()
            trains=[]
            tests=[]
            for d in data:
                train=d.split(" ")[:-1]
                #print(train)
                test=d.split(" ")
                test[-1]=str(int(test[-1]))
                #print(test)
                #break
                trains.append(train)
                tests.append(test)

            file_1=os.getcwd()+"\\"+file[:-4]
            #print(file_1)
            tt=0
            t_t=0
            with open(file_1+"\\train.txt","w+") as ft:
                for t_train in trains:
                    for t_ in t_train[1:]:
                        #print(tt+1,t_)
                        ft.write(str(tt+1)+" "+t_+"\n")
                    tt+=1

            with open(file_1+"\\test.txt","w+") as ft:
                for t_test in tests:
                    for t_ in t_test[1:]:
                        ft.write(str(t_t+1)+" "+t_+"\n")
                    t_t+=1

if __name__ == '__main__':
   dataProcess()







