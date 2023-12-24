if __name__ == '__main__':
    with open("ml-1m.txt","r+") as fr:
        data=fr.readlines()
    m=0
    for d in data:
        temp=d.split(" ")
        m=max(m,len(temp))
    print(m)