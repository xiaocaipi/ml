# !/usr/bin/python
# -*- coding:utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import codecs
import random

infinite = float(-2**31)


def log_normalize(a):
    #先把sum 算出来
    s = 0
    for x in a:
        s += x
    if s == 0:
        print "Error..from log_normalize."
        return
    #对s 取对数
    s = math.log(s)
    for i in range(len(a)):
        #如果 a i 是0 的话 就不能 ln0 ，ln0 是负的无穷打 ，是最小的数
        if a[i] == 0:
            #这里定义一个最小的数infinite
            a[i] = infinite
        else:
            #当前的值 减掉 ln s
            a[i] = math.log(a[i]) - s


def log_sum(a):
    if not a:   # a为空
        return infinite
    m = max(a)
    s = 0
    for t in a:
        s += math.exp(t-m)
    return m + math.log(s)


def calc_alpha(pi, A, B, o, alpha):
    for i in range(4):
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    T = len(o)
    temp = [0 for i in range(4)]
    del i
    for t in range(1, T):
        for i in range(4):
            for j in range(4):
                temp[j] = (alpha[t-1][j] + A[j][i])
            alpha[t][i] = log_sum(temp)
            alpha[t][i] += B[i][ord(o[t])]


def calc_beta(pi, A, B, o, beta):
    T = len(o)
    for i in range(4):
        beta[T-1][i] = 1
    temp = [0 for i in range(4)]
    del i
    for t in range(T-2, -1, -1):
        for i in range(4):
            beta[t][i] = 0
            for j in range(4):
                temp[j] = A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
            beta[t][i] += log_sum(temp)


def calc_gamma(alpha, beta, gamma):
    for t in range(len(alpha)):
        for i in range(4):
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(gamma[t])
        for i in range(4):
            gamma[t][i] -= s


def calc_ksi(alpha, beta, A, B, o, ksi):
    T = len(alpha)
    temp = [0 for x in range(16)]
    for t in range(T-1):
        k = 0
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][ord(o[t+1])] + beta[t+1][j]
                temp[k] =ksi[t][i][j]
                k += 1
        s = log_sum(temp)
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -= s


def bw(pi, A, B, alpha, beta, gamma, ksi, o):
    T = len(alpha)
    for i in range(4):
        pi[i] = gamma[0][i]
    s1 = [0 for x in range(T-1)]
    s2 = [0 for x in range(T-1)]
    for i in range(4):
        for j in range(4):
            for t in range(T-1):
                s1[t] = ksi[t][i][j]
                s2[t] = gamma[t][i]
            A[i][j] = log_sum(s1) - log_sum(s2)
    s1 = [0 for x in range(T)]
    s2 = [0 for x in range(T)]
    for i in range(4):
        print "bw", i
        for k in range(65536):
            valid = 0
            if k % 10000 == 0:
                print "bw - k", k
            for t in range(T):
                if ord(o[t]) == k:
                    s1[valid] = gamma[t][i]
                    valid += 1
                s2[t] = gamma[t][i]
            if valid == 0:
                B[i][k] = infinite
            else:
                B[i][k] = log_sum(s1[:valid]) - log_sum(s2)


def baum_welch(pi, A, B):
    f = file(".\\1.txt")
    sentence = f.read()[3:].decode('utf-8')
    f.close()
    T = len(sentence)
    alpha = [[0 for i in range(4)] for t in range(T)]
    beta = [[0 for i in range(4)] for t in range(T)]
    gamma = [[0 for i in range(4)] for t in range(T)]
    ksi = [[[0 for j in range(4)] for i in range(4)] for t in range(T-1)]
    for time in range(3):
        print "calc_alpha"
        calc_alpha(pi, A, B, sentence, alpha)    # alpha(t,i):给定lamda，在时刻t的状态为i且观测到o(1),o(2)...o(t)的概率
        print "calc_beta"
        calc_beta(pi, A, B, sentence, beta)      # beta(t,i)：给定lamda和时刻t的状态i，观测到o(t+1),o(t+2)...oT的概率
        print "calc_gamma"
        calc_gamma(alpha, beta, gamma)    # gamma(t,i)：给定lamda和O，在时刻t状态位于i的概率
        print "calc_ksi"
        calc_ksi(alpha, beta, A, B, sentence, ksi)    # ksi(t,i,j)：给定lamda和O，在时刻t状态位于i且在时刻i+1，状态位于j的概率
        print "bw"
        bw(pi, A, B, alpha, beta, gamma, ksi, sentence)
        print "time", time
        print "Pi:", pi
        print "A", A


def mle():  # 0B/1M/2E/3S
    #把 B  M  E  S  叫做  0 1 2 3   4个隐状态
    pi = [0] * 4   # npi[i]：i状态的个数
    a = [[0] * 4 for x in range(4)]     # na[i][j]：从i状态到j状态的转移个数
    b = [[0]* 65536 for x in range(4)]  # nb[i][o]：从i状态到o字符的个数
    #人民日报的语料库，有人标注好的
    f = file("./24.pku_training.utf8")
    #把前3个文件头去掉，剩下数据转成utf8
    data = f.read()[3:].decode('utf-8')
    f.close()
    #是有2个空格 进行分开， 分词的，得到词
    tokens = data.split('  ')
    last_q = 2
    iii = 0
    old_progress = 0
    print '进度：'
    for k, token in enumerate(tokens):
        #计算进度
        progress = float(k) / float(len(tokens))
        if progress > old_progress + 0.1:
            print '%.3f' % progress
            old_progress = progress
        #先strip  一下  2边空格去掉
        token = token.strip()
        n = len(token)
        #如果token 的长度 是0，那就不是有效的，就continue 掉
        if n <= 0:
            continue
        #token 长度是1 的话 就表明是一个 single   single 的是隐状态是3
        if n == 1:
            #pi 3  就多了一个
            pi[3] += 1
            #前一个状态 到 3  就多了1次
            a[last_q][3] += 1   # 上一个词的结束(last_q)到当前状态(3S)
            #从3 到 token 0 这个字 就多了1次
            b[3][ord(token[0])] += 1
            #当前状态就是3 了
            last_q = 3
            continue
        # 初始向量
        #如果不是1 的情况下，不是single 的情况下，只要长度不为1  一定有一个 B  和E
        #因此 pi 0  和pi2 都要加1 
        pi[0] += 1
        pi[2] += 1
        # middle 的话 要加上 n-2  比如一个token  是“解放思想” 那么 middle 就是2
        pi[1] += (n-2)
        # 转移矩阵   是从上一个状态 到了0  加上1
        a[last_q][0] += 1
        #当线的状态到 2
        last_q = 2
        #如果等于2的时候  从0 到2  是 加1 了
        if n == 2:
            a[0][2] += 1
        #如果比2还大
        else:
            #从0 到 1 多了1个 
            a[0][1] += 1
            #从1  到1  多了n-3  去掉 B E  和一个个M
            a[1][1] += (n-3)
            #从M 到E 多了一个
            a[1][2] += 1
        # 发射矩阵
        #从0 到 第一个字 多了1
        b[0][ord(token[0])] += 1
        #从 2 到最后一个字 多了1
        b[2][ord(token[n-1])] += 1
        #然后 中间的字 每个都从 1 到 对应的字多了1
        for i in range(1, n-1):
            b[1][ord(token[i])] += 1
    #上面的循环 对 A B pi  都数出来
    # 数出来之后 做正则化 ，取的对数之后再做正则化
    #pi的归一化
    log_normalize(pi)
    for i in range(4):
        #a 和b 的每一行 做归一化
        log_normalize(a[i])
        log_normalize(b[i])
    return [pi, a, b]


def list_write(f, v):
    for a in v:
        f.write(str(a))
        f.write(' ')
    f.write('\n')

#对pi A  B  进行保存
def save_parameter(pi, A, B):
    #保存在 pi   A   B  3个txt  里面去
    f_pi = open(".\\pi.txt", "w")
    list_write(f_pi, pi)
    f_pi.close()
    f_A = open(".\\A.txt", "w")
    for a in A:
        list_write(f_A, a)
    f_A.close()
    f_B = open(".\\B.txt", "w")
    for b in B:
        list_write(f_B, b)
    f_B.close()


if __name__ == "__main__":
    pi, A, B = mle()
    save_parameter(pi, A, B)
    print "训练完成..."
