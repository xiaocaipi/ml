#!/usr/bin/python
# -*- coding:utf-8 -*-

import operator

#n*(n-1)*(n-2)*(n-i+1)  就是n的阶层
def c(n, k):
    #reduce 里面 的操作是乘，从n-k+1  到 n+1  这些数 做连乘操作  这个是分子 
    #分母的画 是从1 到k+1 的连乘
    return reduce(operator.mul, range(n-k+1, n+1)) / reduce(operator.mul, range(1, k+1))

# 参数一个是 次数  一个是正确率  这里p传过来是0.6
#一个分类器能够分类正确的概率是0.6  做10次这样的分类 最终正确的概率，那么就必须要 5 次以上
def bagging(n, p):
    s = 0
	#这里是从5 次到10 次
    for i in range(n / 2 + 1, n + 1):
        s += c(n, i) * p ** i * (1 - p) ** (n - i)
    return s


if __name__ == "__main__":
	#这里的10 是一个偶数   从10 开始  不包括 101  每次递进10 个  t 就是 10  20  到100   做t次采样 正确率 
    for t in range(10, 101, 10):
        print t, '次采样正确率：', bagging(t, 0.6)


