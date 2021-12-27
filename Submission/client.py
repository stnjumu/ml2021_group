# -*- coding: utf-8 -*-

import requests
import glob
import os
from time import sleep

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f :
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def main(ip, port, sid, token, ans, problem):
    print("正在提交...")
    url = "http://%s:%s/jsonrpc" % (ip, port)

    payload = {
        "method": problem,
        "params": [ans],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(
        url,
        json=payload,
        headers={"token": token, "sid": sid}
    ).json()

    print(response)
    if "auth_error" in response:
        print("您的认证信息有误")
        return response["auth_error"]
    elif "error" not in response:
        print("测试完成，请查看分数")
        return response["result"]
    else:
        print("提交文件存在问题，请查看error信息")
        return response["error"]["data"]["message"]


# if __name__ == "__main__":
def submitCarDemand_evaluate(path, studentID, password):
    # 需要修改的参数：problem, sid, token

    # problem 参数：
    #    Action_evaluate:        低分辨率视频行为识别
    #    FoodPredict_evaluate:   菜品分类
    #    StoreSale_evaluate:     商品销售额预测
    #    Toxicity_evaluate:      恶意评论分类
    #    CarDemand_evaluate:     汽车需求量预测
    #    FineGrainedCar_evaluate:细粒度汽车分类
    #    Traffic_evaluate:       疫情人流量预测
    #    Mask_evaluate:          口罩检测

    problem = "FoodPredict_evaluate"
    # IP 固定为 115.236.52.125
    ip = "115.236.52.125"
    # 端口不需要修改
    port = "4000"
    # 改成你的学号
    sid = studentID
    # 改成你的口令
    token = password
    
    if problem in ["Action_evaluate",
                'FoodPredict_evaluate',
                'StoreSale_evaluate',
                'Toxicity_evaluate',
                'CarDemand_evaluate',
                'FineGrainedCar_evaluate',
                'Traffic_evaluate']:
        with open(path) as f:
            d = list(f.readlines())
    elif problem == "Mask_evaluate":
        submit_dir = './submission'
        submissions = os.listdir('./submission')
        d = {}
        for submit in submissions:
            submit_file = os.path.join(submit_dir,submit)
            with open(submit_file,'r') as f:
                d[submit] = f.read().splitlines()

    score = main(ip, port, sid, token, d, problem)
    return score
