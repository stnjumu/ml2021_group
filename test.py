from ModelLib.swin import SwinNet
from ModelLib.tresnet_v2 import TResnetL_V2
from ModelLib.efficientnet_v2 import *
from UtilLib.Read_annos_mat import read_annos_to_np
from ModelLib.Model1 import Model1
from ModelLib.resnet101 import RenNet101_head
from ModelLib.resnet import resnet50
from DatasetLib.Dataset1 import Dataset1
from DatasetLib.DatasetTorch import DatasetTorch
import UtilLib.logger as Logger

import torch
import torch.utils.data as torchData
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import logging
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="4"
# 参数字典
paramDict = {
     # 训练设置
    'batch_size': 12, 
    'learning_rate': 3e-4,
    'epoches': 100, 
    'loss': torch.nn.CrossEntropyLoss(),

    # 模型设置
    'model': TResnetL_V2(num_classes=196), # 自定义模型
    'resume_training': True, # 继续训练
    'checkpointName': 'stanford_cars_tresnet-l-v2_96_27.pth', # 检查点名称
    'ignore_optim_flag': False, # 忽略部分预训练模型参数
    'ignore_backbone_name': 'backbone', # 要忽略的预训练参数名称
    
    # 数据集设置
    'dataset': DatasetTorch, # 自定义数据库

     # 日志设置
    'datasetDir': './dataset/', # 数据集存放路径
    'checkpointDir':  './checkpoint/', # 检查点路径
    'log_path': 'log', # 日志路径
    'val_freq' : 2, # 每隔epoch验证
    'print_freq' : 100, # 每隔step计算准确率
    'save_freq' : 10, # 每隔epoch存储模型，暂未使用
}

# 一些配置
torch.set_default_tensor_type('torch.FloatTensor')
# torch.cuda.set_device(4)
datasetDir = paramDict['datasetDir'] # 数据集文件夹
checkpointDir = paramDict['checkpointDir'] # 创建保存checkpoint的文件夹
os.makedirs(checkpointDir, exist_ok=True)

# 日志配置
log_path = paramDict['log_path']
Logger.setup_logger('test', log_path, '{}_test'.format(datetime.now().strftime('%y%m%d_%H%M')), level=logging.INFO, screen=True)
logger_test = logging.getLogger('test')
for k,v in paramDict.items():
    logger_test.info("{} = {}".format(k, v))

# 超参数
batch_size = paramDict['batch_size']
learning_rate = paramDict['learning_rate']
epoches = paramDict['epoches']

# 训练参数
val_freq = paramDict['val_freq']
print_freq = paramDict['print_freq']
save_freq = paramDict['save_freq']


# 模型
model = paramDict['model']
if paramDict['resume_training']:
    checkpoint = torch.load(os.path.join(checkpointDir , paramDict['checkpointName']), map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    logger_test.info(
            'Load model from {}/{}.'.format(checkpointDir, paramDict['checkpointName']))
model = model.cuda() # 模型放GPU上；

# 数据集
cars_test_annos_Path = os.path.join(datasetDir, 'cars_test_annos.mat')
img_dir = os.path.join(datasetDir, 'cars_test')
data_test = read_annos_to_np(cars_test_annos_Path)
dataset = paramDict['dataset'](img_dir, data_test, split='test') 
testLoader = torchData.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
# 测试并输出
model.eval()
output_name = datetime.now().strftime('%y%m%d_%H%M')
with torch.no_grad():
    label_all = []
    label_predict_all = []
    pbar = tqdm(testLoader)
    start = time.time()
    for data in pbar:
        imgs, file_name = data
        imgs= imgs.cuda()
        labels_predict = model(imgs)
        labels_predict = np.argmax(labels_predict.cpu().numpy(), axis = 1)
        with open('log/{}_submission.txt'.format(output_name), 'a+') as f:
            for i in range(labels_predict.shape[0]):
                print(file_name[i], labels_predict[i]+1, file=f)
    logger_test.info(
            'Finished, Time cost is {}.'.format(time.time()-start))

 