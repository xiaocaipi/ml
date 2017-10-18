# -*- coding:utf-8 -*-
a = "@小黑豆 [AT]87678[UID]"
# print a

import re

# 将正则表达式编译成Pattern对象
pattern = re.compile(r'@.+? \[AT].+?\[UID]')

# 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
match = pattern.match('@小黑豆 [AT]87678[UID] asdasfa @小黑豆 [AT]87678[UID]')

# if match:
#     # 使用Match获得分组信息
#     print match.group(0)


for m in pattern.finditer('@小黑豆 [AT]87678[UID] asdasfa @小黑豆 [AT]87678[UID]'):
    print m.group()