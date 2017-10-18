# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#在一个T 周期 取多少样本 size  
#T 是一共取多少个周期
def triangle_wave(size, T):
    #从-1 到1  size 个数  特意不要最后一个
    t = np.linspace(-1, 1, size, endpoint=False)
    # where
    # y = np.where(t < 0, -t, 0)
    # y = np.where(t >= 0, t, y)
    #直接对t 取一个绝对值
    y = np.abs(t)
    #取绝对值 之后重复诺干次  再去减去0.5
    y = np.tile(y, T) - 0.5
    x = np.linspace(0, 2*np.pi*T, size*T, endpoint=False)
    return x, y


def sawtooth_wave(size, T):
    t = np.linspace(-1, 1, size)
    y = np.tile(t, T)
    x = np.linspace(0, 2*np.pi*T, size*T, endpoint=False)
    return x, y


def triangle_wave2(size, T):
    x, y = sawtooth_wave(size, T)
    return x, np.abs(y)


def non_zero(f):
    f1 = np.real(f)
    f2 = np.imag(f)
    eps = 1e-4
    return f1[(f1 > eps) | (f1 < -eps)], f2[(f2 > eps) | (f2 < -eps)]


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    #数据不要显示指数型，显示成小数型
    np.set_printoptions(suppress=True)

    #从0 到2 Pi  的等差数列  一共取16个数字，不要终点的值
    x = np.linspace(0, 2*np.pi, 16, endpoint=False)
    print '时域采样值：', x
    #这里做一个 sin（2x）  和sin（3x）  这里加了 四分之pi   频率是一样的  只是相位不一样  
    #然后这2个加在一起
    y = np.sin(2*x) + np.sin(3*x + np.pi/4)
    #当然也可以得到一个标准的 sinx
    # y = np.sin(x)

    #N 是采样点的个数
    N = len(x)
    print '采样点个数：', N
    print '\n原始信号：', y
    #对y 进行快速傅立叶变换  把时域的信号输入进去 出来一个频域的信号 f
    f = np.fft.fft(y)
    # 频域的信号f 处以 N 就是频域信号的值   这个值是实数 加需数的  有实部 和需部
    print '\n频域信号：', f/N
    #然后对这个值取模
    a = np.abs(f/N)
    print '\n频率强度：', a
    #对这个信号进行逆傅立叶变换  得到iy
    iy = np.fft.ifft(f)
    #这个iy 的需部都是0，因为原始信号是没有需部的
    print '\n逆傅里叶变换恢复信号：', iy
    #取出需部
    print '\n虚部：', np.imag(iy)
    #获取实部
    print '\n实部：', np.real(iy)
    #利用 allclose  判断2个 部分 是不是几乎相同的
    print '\n恢复信号与原始信号是否相同：', np.allclose(np.real(iy), y)

    #做一个2行1列 对第一个图像进行绘制
    plt.subplot(211)
    #把x y 本身画进去
    plt.plot(x, y, 'go-', lw=2)
    plt.title(u'时域信号', fontsize=15)
    plt.grid(True)
    #做一个2行1列 对第二个图像进行绘制
    plt.subplot(212)
    w = np.arange(N) * 2*np.pi / N
    print u'频率采样值：', w
    #stem 用柱子
    plt.stem(w, a, linefmt='r-', markerfmt='ro')
    plt.title(u'频域信号', fontsize=15)
    plt.grid(True)
    plt.show()

    # 三角/锯齿波
    x, y = triangle_wave(20, 5)
    # x, y = sawtooth_wave(20, 5)
    N = len(y)
    f = np.fft.fft(y)
    # print '原始频域信号：', np.real(f), np.imag(f)
    print '原始频域信号：', non_zero(f)
    a = np.abs(f / N)

    # np.real_if_close
    f_real = np.real(f)
    eps = 0.1 * f_real.max()
    print eps
    f_real[(f_real < eps) & (f_real > -eps)] = 0
    f_imag = np.imag(f)
    eps = 0.1 * f_imag.max()
    print eps
    f_imag[(f_imag < eps) & (f_imag > -eps)] = 0
    f1 = f_real + f_imag * 1j
    y1 = np.fft.ifft(f1)
    y1 = np.real(y1)
    # print '恢复频域信号：', np.real(f1), np.imag(f1)
    print '恢复频域信号：', non_zero(f1)

    plt.figure(figsize=(8, 8), facecolor='w')
    plt.subplot(311)
    plt.plot(x, y, 'g-', lw=2)
    plt.title(u'三角波', fontsize=15)
    plt.grid(True)
    plt.subplot(312)
    w = np.arange(N) * 2*np.pi / N
    plt.stem(w, a, linefmt='r-', markerfmt='ro')
    plt.title(u'频域信号', fontsize=15)
    plt.grid(True)
    plt.subplot(313)
    plt.plot(x, y1, 'b-', lw=2, markersize=4)
    plt.title(u'三角波恢复信号', fontsize=15)
    plt.grid(True)
    plt.tight_layout(1.5, rect=[0, 0.04, 1, 0.96])
    plt.suptitle(u'快速傅里叶变换FFT与频域滤波', fontsize=17)
    plt.show()
