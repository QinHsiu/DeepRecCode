
def isAdditiveNumber(num: str) -> bool:
    n = len(num)
    # 长度小于3的必定不是累加数
    if n < 3:
        return False

    # 定义函数
    # 输入前面的两个数字 判断与
    def check(p1, p2, j):
        while j < n:
            # 将两者之和转换为string类型
            p = str(int(p1) + int(p2))
            # 判断后续字符串的是否与当前计算结果相等
            # 长度从j开始
            if j + len(p) > n:
                return False
            if num[j: j + len(p)] != p:
                return False
            # 记录当前的位置
            j += len(p)
            # 更新当前需要计算的两个数
            p1, p2 = p2, p
        # 如果循环顺利结束 说明是可行的
        return True

    if num[0] == "0":
        p1 = num[0]
        for j in range(2, n):
            p2 = num[1: j]
            if check(p1, p2, j):
                return True
        return False
    else:
        for i in range(1, n // 2 + 1):
            if num[i] == "0":
                p1 = num[: i]
                p2 = 0
                if check(p1, p2, i + 1):
                    return True
                else:
                    continue
            for j in range(i + 1, n):
                p1 = num[: i]
                p2 = num[i: j]
                if check(p1, p2, j):
                    return True
        return False

print(isAdditiveNumber("199001200"))