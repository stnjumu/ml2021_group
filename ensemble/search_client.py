# -*- coding: utf-8 -*-

import requests
import glob
import os
import numpy as np

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f :
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def submit(ip, port, sid, token, ans, problem):
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

from itertools import combinations
import time
if __name__ == "__main__":
    # 需要修改的参数：problem, sid, token
    name_arr = []
    # files = list(glob.glob('*_*.txt'))
    files = ['Aug1_EfficientNetL_384.txt', 'Aug1_Norm.txt', 'Aug2_EfficientNetL_468.txt', 'Aug2_ResNest_L200.txt', 'ResNest269E35.txt']
    N = len(files)
    for len in range(6,N+1):
        new_files_arr = list(combinations(files, len))
        for new_files in new_files_arr:
            result_arr = np.zeros((8041, 196))
            
            for file in new_files:
                with open(file, 'r') as result:
                    for idx, line in enumerate(result, start=0):
                        name, id = line.split(' ')
                        if idx < 8041:
                            result_arr[idx][int(id)-1] += 1
                        name_arr.append(name)
            name_arr = name_arr[:8041]
            result_idx = result_arr.argmax(axis=1)
            with open('submission.txt', 'w') as f:
                for name, id in zip(name_arr, result_idx):
                    print(name, id+1, file=f)
            
            problem = "FineGrainedCar_evaluate"
            # IP 固定为 115.236.52.125
            ip = "115.236.52.125"
            # 端口不需要修改
            port = "4000"
            # 改成你的学号
            sid = "TEAM_43"
            # 改成你的口令
            token = "223455"
            with open("submission.txt") as f:
                d = list(f.readlines())

            score = submit(ip, port, sid, token, d, problem)
            with open('log.txt', 'a+') as f:
                print(score, file=f, end=', ')
            with open('log.txt', 'a+') as f:
                print(new_files, file=f)
            time.sleep(10)
