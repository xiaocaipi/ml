# coding: utf-8

# In[1]:

# 验证码生成库  用的是captcha  这个库
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
# PIL  是处理图像的包，也可以生成验证码
from PIL import Image
import random
import sys

# 定义验证码的字符  可以用数字  小写 字符  大写字符，这里主要讲的是数字，大小写的字符也是可以的
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# 验证码的 长度是4位，4位的话 从0000 到9999  只有1万中可能
# 这里用的是number 数字  如果要用 数字+小写字母的话  可以是char_set=number+alphabet
def random_captcha_text(char_set=number, captcha_size=4):
    # 验证码列表
    captcha_text = []
    for i in range(captcha_size):
        # 随机选择，这里用随机数来产生验证码
        c = random.choice(char_set)
        # 加入验证码列表
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    # 生成验证码对象  图片是160 *60的
    image = ImageCaptcha()
    # 获得随机生成的验证码
    captcha_text = random_captcha_text()
    # 把验证码列表转为字符串
    captcha_text = ''.join(captcha_text)
    # 生成验证码
    captcha = image.generate(captcha_text)
    image.write(captcha_text, '../data/captcha/images/' + captcha_text + '.jpg')  # 写到文件


# 数量少于10000，因为重名
num = 10000
if __name__ == '__main__':
    # 做10000  个循环，会生成1万个 验证码的图片，但是会少于1万，因为会有重名
    # 因为第1 次比如生成一个 1234  这个文件 下次再生成1234  的时候会去覆盖
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i + 1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    print("生成完毕")


# In[ ]:




# In[ ]:



