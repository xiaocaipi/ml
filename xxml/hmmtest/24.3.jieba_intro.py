# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import jieba
import jieba.posseg

#用jieba 分词
#这个分词效果 要比自己写的好一点，但不见得 都对
if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    f = open('./24.novel.txt')
    str = f.read().decode('utf-8')
    f.close()

    seg = jieba.posseg.cut(str)
    for s in seg:
        print s.word, s.flag, '|',
        # print s.word, '|',
