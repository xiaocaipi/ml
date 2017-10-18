# -*- coding:utf-8 -*-
import requests
import json,time
import signature

url = "http://192.168.100.69:8080"
headers = {"Content-Type": "application/json"}
apiKey = "a6b9e216b8be4a37bd9e9aee92ef5bb8"
timestamp = "1491466356433"

def addsign():
    startTime = time.time() * 1000
    global apiKey
    global timestamp
    global url
    global headers
    title = "新增签名"
    test_url = url+"/msgweb/msg/sign/add/v1.0.0"
    reqJson = '{"operator":"12","sign":"非常干脆2"}'
    reqJson1 = {"operator":"12","sign":"非常干脆2"}
    signature1 = signature.getsignature(reqJson)#将需要加签的数据传给加签方法
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    status_code = getcode_res.status_code
    return_data= getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is False and errMsg == "添加签名失败---原因：供应商[云片,签名已存在]添加签名失败":
        success = 'Pass'
    else:
        success = 'False'
    endTime = time.time() * 1000
    print "接口调用时间%.2f" % (endTime - startTime)
    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1,  'return_data': return_data, 'success': success}

def updatesign():
    global apiKey
    global timestamp
    global url
    global headers
    title = '修改签名'
    test_url = url+"/msgweb/msg/sign/update/v1.0.0"
    reqJson = '{"signId":"40","operator":"12","sign":"好来多"}'
    reqJson1 = {"signId":"40","operator":"12","sign":"好来多"}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is False and errMsg == "修改签名失败---原因：供应商[云片,通过审核的签名不可修改]修改签名失败":
        success = 'Pass'
    else:
        success = 'False'
    #print success,errMsg
    #print return_data,errMsg
    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def addtemplate():
    global apiKey
    global timestamp
    global url
    global headers
    title = '新增模板'
    test_url = url+"/msgweb/msg/template/add/v1.0.0"
    reqJson = '{"appId": "10003","signId": "29","operator":"12","title":"旺铺APP","templateType":"2","templateContent":"新品上市"}'
    reqJson1 = {"appId": "10003","signId": "29","operator":"12","title":"旺铺APP","templateType":"2","templateContent":"新品上市"}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is False and errMsg == "添加模板失败----原因：供应商[云片,参数 tpl_content 格式不正确，tpl_content模板内容已存在,重复模板ID:1879398]模板添加失败！":
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def updatetemplate():
    global apiKey
    global timestamp
    global url
    global headers
    title = '编辑模板'
    test_url = url+"/msgweb/msg/template/update/v1.0.0"
    reqJson = '{"appId": "10003","signId": "29","operator":"12","templateType":"1","templateId":"M000027","title":"这是标题","templateContent":"您的邀请码是：{0}，请勿转发给其他人员"}'
    reqJson1 = {"appId": "10003","signId": "29","operator":"12","templateType":"1","templateId":"M000027","title":"这是标题","templateContent":"您的邀请码是：{0}，请勿转发给其他人员"}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is False :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def deltemplate():
    global apiKey
    global timestamp
    global url
    global headers
    title = '删除模板'
    test_url = url+"/msgweb/msg/template/delete/v1.0.0"
    reqJson = '{"appId": "10003","operator":"112","templateId":"M000024"}'
    reqJson1 = {"appId": "10003","operator":"112","templateId":"M000024"}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is False :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def syncTemplate():
    global apiKey
    global timestamp
    global url
    global headers
    title = '同步模板审核状态'
    test_url = url+"/msgweb/msg/template/syncTemplate/v1.0.0"
    reqJson = '{"appId":"10003","templateId":"M000038","operator":"100"}'
    reqJson1 = {"appId":"10003","templateId":"M000038","operator":"100"}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is True :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def syncSign():
    global apiKey
    global timestamp
    global url
    global headers
    title = '同步签名审核状态'
    test_url = url+"/msgweb/msg/sign/syncSign/v1.0.0"
    reqJson = '{"sign":"好来多","operator":"99"}'
    reqJson1 = {"sign":"好来多","operator":"99"}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is True :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def singlemessage():
    global apiKey
    global timestamp
    global url
    global headers
    title = '短信单条发送'
    test_url = url+"/msgweb/msg/message/send/v1.0.0"
    reqJson = '{"appId":"10003","businessId":"10036","operator":"12","mobile":"18659206198","environment":"1","paramter":{"sms":"12|12"}}'
    reqJson1 = {"appId":"10003","businessId":"10036","operator":"12","mobile":"18659206198","environment":"1","paramter":{"sms":"12|12"}}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is True :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def mutimessage():
    global apiKey
    global timestamp
    global url
    global headers
    title = '短信多条发送'
    test_url = url+"/msgweb/msg/message/mutiSend/v1.0.0"
    reqJson = '{"appId":"10003","businessId":"10036","operator":"12","mobile":"18659206198,18659206192","environment":"0","paramter":{"sms":"12|77"}}'
    reqJson1 = {"appId":"10003","businessId":"10036","operator":"12","mobile":"18659206198,18659206192","environment":"0","paramter":{"sms":"12|77"}}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is True :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def singlenews():
    global apiKey
    global timestamp
    global url
    global headers
    title = '消息单条发送'
    test_url = url+"/msgweb/msg/message/send/v1.0.0"
    reqJson = '{"appId":"10003","businessId":"10038","operator":"12","title":"这里是标题423","deviceType":"android","notifyChannel":"jpush","bussParam":"{\"tcd\":\"333\"}","url":"www.sososo.com","type":"0","deviceId":"18171adc033ba8295cb","environment":"1","paramter":{"notify":"xiaoliao|9098|899|ddd"}}'
    reqJson1 = {"appId":"10003","businessId":"10038","operator":"12","title":"这里是标题423","deviceType":"android","notifyChannel":"jpush","bussParam":"{\"tcd\":\"333\"}","url":"www.sososo.com","type":"0","deviceId":"18171adc033ba8295cb","environment":"1","paramter":{"notify":"xiaoliao|9098|899|ddd"}}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is True :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def mutinews():
    global apiKey
    global timestamp
    global url
    global headers
    title = '消息多条发送'
    test_url = url+"/msgweb/msg/message/mutiSend/v1.0.0"
    reqJson = '{"appId":"10003","businessId":"10040","operator":"12","deviceInfo":"jpush|ios|18171adc033ba8295cb,jpush|android|160a3797c835e8bfea5","bussParam":"{\"tcd\":\"333\"}","title":"今天星期二999","url":"www.sososo.com","type":"0","environment":"0","paramter":{"notify":"12|322|3434|334"}}'
    reqJson1 = {"appId":"10003","businessId":"10040","operator":"12","deviceInfo":"jpush|ios|18171adc033ba8295cb,jpush|android|160a3797c835e8bfea5","bussParam":"{\"tcd\":\"333\"}","title":"今天星期二999","url":"www.sososo.com","type":"0","environment":"0","paramter":{"notify":"12|322|3434|334"}}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is True :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def mutisend():
    global apiKey
    global timestamp
    global url
    global headers
    title = '消息+短信多条发送'
    test_url = url+"/msgweb/msg/message/mutiSend/v1.0.0"
    reqJson = '{"appId":"10003","businessId":"10041","operator":"12","deviceInfo":"jpush|android|160a3797c835e8bfea5,jpush|ios|18171adc033ba8295cb","bussParam":"{\"tcd\":\"333\"}","title":"今天星期二","url":"www.123sososo.com","type":"0","mobile":"18659206199","environment":"1","paramter":{"sms":"","notify":""}}'
    reqJson1 = {"appId":"10003","businessId":"10041","operator":"12","deviceInfo":"jpush|android|160a3797c835e8bfea5,jpush|ios|18171adc033ba8295cb","bussParam":"{\"tcd\":\"333\"}","title":"今天星期二","url":"www.123sososo.com","type":"0","mobile":"18659206199","environment":"1","paramter":{"sms":"","notify":""}}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is True :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}

def singlesend():
    global apiKey
    global timestamp
    global url
    global headers
    title = '消息+短信单条发送'
    test_url = url+"/msgweb/msg/message/send/v1.0.0"
    reqJson = '{"appId": "10003","businessId": "10042","operator": "12","deviceType": "android","notifyChannel": "jpush","bussParam": "{\"tcd\":\"333\"}","url": "www.sososo.com","deviceId": "18171adc033ba8295cb","title": "今天星期五988998","mobile": "15676722403","type":"0","environment":"0","paramter": {"sms": "22|322","notify": "6556"}}'
    reqJson1 = {"appId": "10003","businessId": "10042","operator": "12","deviceType": "android","notifyChannel": "jpush","bussParam": "{\"tcd\":\"333\"}","url": "www.sososo.com","deviceId": "18171adc033ba8295cb","title": "今天星期五988998","mobile": "15676722403","type":"0","environment":"0","paramter": {"sms": "22|322","notify": "6556"}}
    signature1 = signature.getsignature(reqJson)
    form_data = {"apiKey":apiKey,"timestamp":timestamp,"signature":str(signature1),"reqJson":reqJson1}
    form_data1 = json.dumps(form_data,ensure_ascii=False)
    getcode_res = requests.post(test_url, data=form_data1, headers=headers)
    return_data = getcode_res.json()
    errMsg = return_data["errMsg"]
    success = return_data["success"]
    if success is True :
        success = 'Pass'
    else:
        success = 'False'

    return {'title': title, 'test_url': test_url,'reqJson': reqJson, 'reqJson1': reqJson1,'signature1':signature1, 'return_data': return_data,'success': success}


if __name__ == "__main__":
    print "aa"
    #addsign()
    #updatesign()