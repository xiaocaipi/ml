# -*- coding:utf-8 -*-
import requests
import json,time
import signature

url = "http://192.168.100.69:8080"
headers = {"Content-Type": "application/json"}
apiKey = "a6b9e216b8be4a37bd9e9aee92ef5bb8"
timestamp = "1491466356433"

def updatetemplate():
    global apiKey
    global timestamp
    global url
    global headers
    title = '编辑模板'
    test_url = url+"/msgweb/msg/template/update/v1.0.0"
    reqJson = '{"appId": "10003","signId": "29","operator":"12","templateType":"1","templateId":"M000027","title":"123","templateContent":"123"}'
    reqJson1 = {"appId": "10003","signId": "29","operator":"12","templateType":"1","templateId":"M000027","title":"123","templateContent":"123"}
    signature1 = signature.getsignature(reqJson)
    print signature1
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers={"Content-Type": "application/json"})
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    print errMsg
    success = return_data["success"]
    print return_data
    print form_data1
    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}
if __name__ == "__main__":
    updatetemplate()
