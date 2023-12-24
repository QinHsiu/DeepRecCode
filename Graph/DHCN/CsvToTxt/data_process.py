import pickle

def tackle_data(path):
    data_train, groundtruth_train = pickle.load(open(path+"\\train.txt", 'rb'))
    data_test, groundtruth_test = pickle.load(open(path+"\\test.txt", 'rb'))
    length_train = len(data_train)
    length_test=len(data_test)
    print(length_train,length_test)
    for mode in ["crop","pad"]:
        temp_train_data=[]
        temp_test_data=[]
        temp_train_g=[]
        temp_test_g=[]
        with open(path+"//train_"+mode+".txt", "w+") as f:
            if mode=="crop": #剪切
                for it in range(length_train):
                    if len(data_train[it])<1:
                        continue
                    temp_train_data.append(data_train[it])
                    temp_train_g.append(groundtruth_train[it])

                for i in range(len(temp_train_data)):
                    f.write(str(i + 1) + " " + " ".join(map(str, temp_train_data[i])) + " " + str(temp_train_g[i]) + "\n")

            else: #填充
                for it in range(len(data_train)):
                    while len(data_train[it])<1:
                        data_train[it].insert(0,0)

                for i in range(len(data_train)):
                    f.write(str(i) + " " + " ".join(map(str, data_train[i])) + " " + str(groundtruth_train[i]) + "\n")


        with open(path+"//test_"+mode+".txt","w+") as f:
            if mode=="crop":
                for it in range(length_test):
                    if len(data_train[it])<1:
                        continue
                    temp_test_data.append(data_test[it])
                    temp_test_g.append(groundtruth_test[it])

                for j in range(len(temp_test_data)):
                    f.write(str(j) + " " + " ".join(map(str, temp_test_data[j])) + " " + str(temp_test_g[j]) + "\n")

            else: #填充
                for jt in range(len(data_test)):
                    while len(data_test[jt]) < 1:
                        data_test[jt].insert(0, 0)

                for j in range(len(data_test)):
                    f.write(str(j) + " " + " ".join(map(str, data_test[j])) + " " + str(groundtruth_test[j]) + "\n")


if __name__ == '__main__':
    import os
    path=os.getcwd()+"\\DHCN"
    filenames = os.listdir(path)
    for f in filenames:
        t_path=path+"\\"+f
        #print(t_path)
        tackle_data(t_path)
