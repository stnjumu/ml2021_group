# -*- coding: utf-8 -*-

import requests
def show_leaderboard(rank):
    print('{}\t|{}\t|{}\t|{}'\
          .format('sid' + ' ' * 10,
                  'name' + ' ' * 10,
                  'score' + ' ' * 10,
                  'rank' + ' ' * 10))
    team_rank = 0
    for _, line in enumerate(rank):
        sid, name, score, rank = line[0], line[1], line[2], line[3]
        if 'TEAM' not in sid:
            continue
        sid = sid + ' ' * (13 - len(sid))
        name = name + ' ' * (10 - len(name))
        if type(score) == float:
            score = '%.6f' % score
        else:
            score = str(score)
        score = score + ' ' * (15 - len(score))
        line = '{} \t|{}\t|{}\t|{}' \
            .format(sid, name, score, team_rank)
        print(line)
        team_rank += 1

def main(ip, port, sid, token, problem):
    url = "http://%s:%s/jsonrpc" % (ip, port)

    payload = {
        "method": "leaderboard",
        "params": [problem],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(
        url,
        json=payload,
        headers={"token": token, "sid": sid}
    ).json()

    if "auth_error" in response:
        print("您的认证信息有误")
        print(response["auth_error"])
    elif "error" not in response:
        print("请查看结果")
        rank = response['result']
        if type(rank) != str:
            show_leaderboard(rank)
        else:
            print(rank)
    else:
        print("提交存在问题，请查看error信息")
        print(response["error"]["data"]["message"])


if __name__ == "__main__":
    # 需要修改的参数：problem, name

    # problem 参数：
    #    QY_evaluate: 测试
    #    Action_evaluate:        低分辨率视频行为识别
    #    FoodPredict_evaluate:   菜品分类
    #    StoreSale_evaluate:     商品销售额预测
    #    Toxicity_evaluate:      恶意评论分类
    #    CarDemand_evaluate:     汽车需求量预测
    #    FineGrainedCar_evaluate:细粒度汽车分类
    #    Traffic_evaluate:       疫情人流量预测
    #    Mask_evaluate:          口罩检测    

    problem = "FineGrainedCar_evaluate"
    # IP 固定为 115.236.52.125
    ip = "115.236.52.125"
    # 端口不需要修改
    port = "4000"
    # 改成你的学号
    sid = "ZY2106345"
    # 改成你的口令
    token = "gagaga"

    main(ip, port, sid, token, problem)
