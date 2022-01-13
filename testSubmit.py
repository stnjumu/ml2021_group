from ModelLib.swin import SwinNet
from ModelLib.tresnet_v2 import TResnetL_V2
from ModelLib.efficientnet_v2_pretrain import *
from UtilLib.Read_annos_mat import read_annos_to_np
from DatasetLib.DatasetTorch import DatasetTorch
import UtilLib.logger as Logger

import torch
import torch.utils.data as torchData
import numpy as np
from tqdm import tqdm

import time
import timm
import os
import logging
from datetime import datetime
import requests
import argparse


def submit(ip, port, sid, token, ans, problem):
    print("正在提交...", end=' ')
    url = "http://%s:%s/jsonrpc" % (ip, port)

    payload = {
        "method": problem,
        "params": [ans],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(url,
                             json=payload,
                             headers={
                                 "token": token,
                                 "sid": sid
                             }).json()

    if "auth_error" in response:
        print("您的认证信息有误")
        return response["auth_error"]
    elif "error" not in response:
        print("测试完成，请查看分数")
        return response["result"]
    else:
        print("提交文件存在问题，请查看error信息")
        return response["error"]["data"]["message"]


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # 参数字典
    paramDict = {
        # 训练设置
        'batch_size': 12,
        'model': timm.create_model('resnest200e',
                                   pretrained=True,
                                   num_classes=196),  # 自定义模型
        'dataset': DatasetTorch,  # 自定义数据库
        'datasetDir': './dataset/',  # 数据集存放路径
        'checkpointPath': args.model,  # 检查点路径
        'resultDir': args.path  # 预测文件目录
    }

    # torch.cuda.set_device(4)
    datasetDir = paramDict['datasetDir']  # 数据集文件夹
    logPath = '{}/submission_{}.txt'.format(
        paramDict["resultDir"],
        datetime.now().strftime('%y%m%d_%H%M'))
    checkpointPath = paramDict["checkpointPath"]

    # 模型
    model = paramDict['model']
    checkpoint = torch.load(checkpointPath, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.cuda()  # 模型放GPU上；

    # 数据集
    cars_test_annos_Path = os.path.join(datasetDir, 'cars_test_annos.mat')
    img_dir = os.path.join(datasetDir, 'cars_test')
    data_test = read_annos_to_np(cars_test_annos_Path)
    dataset = paramDict['dataset'](img_dir, data_test, split='test')
    testLoader = torchData.DataLoader(dataset,
                                      batch_size=paramDict['batch_size'],
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=8,
                                      drop_last=False)
    # 测试并输出
    model.eval()
    with torch.no_grad():
        pbar = tqdm(testLoader)
        for data in pbar:
            imgs, file_name = data
            imgs = imgs.cuda()
            labels_predict = model(imgs)
            labels_predict = np.argmax(labels_predict.cpu().numpy(), axis=1)
            with open(logPath, 'a+') as f:
                for i in range(labels_predict.shape[0]):
                    print(file_name[i], labels_predict[i] + 1, file=f)

    if args.submit:
        problem = "FineGrainedCar_evaluate"
        ip = "115.236.52.125"
        port = "4000"
        sid = "SY2106335"
        token = "123456"
        with open(logPath) as f:
            d = list(f.readlines())
        score = submit(ip, port, sid, token, d, problem)
        print('Result:[{}], File:[{}], Checkpoint:[{}]'.format(
            score, logPath, checkpointPath))
        # print('Result: ', score, 'Submission Txt Path:[', logPath, 'CheckPoint Path:', checkpointPath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rank")

    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('-p', '--path', type=str)  # submission.txt 存放目录
    parser.add_argument('-s', '--submit', type=bool,
                        default=True)  # submission.txt 存放目录
    parser.add_argument('-gpu', '--gpu_id', type=int, default=0)
    args = parser.parse_args()
    run(args)
