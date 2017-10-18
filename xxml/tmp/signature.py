# -*- coding:utf-8 -*-
#
import requests
import json
import MessageCenter
#实现加签并返回加签的值
def getsignature(reqJson):#从MessageCenter中获取每个接口的reqJson
    key = MessageCenter.apiKey
    iv = MessageCenter.timestamp
    test_url = "http://192.168.201.84:8080/encryweb/encry"
    form_data = {"key":key,"iv":iv,'str':reqJson}
    form =json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url,data=form_data,headers={"Content-Type":"application/x-www-form-urlencoded"})
    getsignature = getcode_res.text
    print getsignature
    return getsignature


if __name__ == "__main__":
    getsignature()
