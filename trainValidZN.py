from torch._C import ParameterDict
from torch.utils.data import dataloader
from UtilLib.Read_annos_mat import read_annos_to_np
from ModelLib.resnet50 import ResNet50
from ModelLib.resnet101 import ResNet101
from DatasetLib.DatasetTorch import DatasetTorch
import UtilLib.logger as Logger

import torch
import torch.utils.data as torchData
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from datetime import datetime

# 参数字典
paramDict = {
     # 训练设置
    'batch_size': 32, 
    'learning_rate': 1e-3,
    'epoches': 400, 
    'loss': torch.nn.CrossEntropyLoss(),
    'multi_gpu': True,
    'gpu_idx': [0, 1, 2, 3],

    # 模型设置
    'model': ResNet101(pretrained = True), # 自定义模型 是否使用预训练模型
    'resume_training': False, # 继续训练
    'checkpointName': 'checkpoint_epoch10.pth', # 检查点名称
    'ignore_optim_flag': False, # 忽略部分预训练模型参数
    'ignore_backbone_name': 'backbone', # 要忽略的预训练参数名称
    
    # 数据集设置
    'dataset': DatasetTorch, # 自定义数据库
    'augment': True, # 训练集数据增强,
    'img_size': [300,300], # 图片缩放至尺寸
    'img_mode': "RGB", #图片颜色模式 "RGB" = RGB, "L" = 灰度图 

     # 日志设置
    'datasetDir': './dataset/', # 数据集存放路径
    'checkpointDir':  './checkpoint/', # 检查点路径
    'log_path': 'log', # 日志路径
    'val_freq' : 1, # 每隔epoch验证
    'print_freq' : 100, # 每隔step计算准确率
    'save_freq' : 10, # 每隔epoch存储模型，暂未使用
}

# 一些配置
torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(0)
datasetDir = paramDict['datasetDir'] # 数据集文件夹
checkpointDir = paramDict['checkpointDir'] # 创建保存checkpoint的文件夹
os.makedirs(checkpointDir, exist_ok=True)

""" multi_gpu = paramDict['multi_gpu']
gpu_idx = paramDict['gpu_idx']
if multi_gpu:
    dist.init_process_group(backend='nccl') """

# 日志配置
log_path = paramDict['log_path']
Logger.setup_logger(None,log_path,
                    '{}_train'.format(datetime.now().strftime('%y%m%d_%H%M')), level=logging.INFO, screen=True)
Logger.setup_logger('val',log_path, '{}_val'.format(datetime.now().strftime('%y%m%d_%H%M')), level=logging.INFO)
logger_base = logging.getLogger('base')
logger_val = logging.getLogger('val')

# 超参数
batch_size = paramDict['batch_size']
learning_rate = paramDict['learning_rate']
epoches = paramDict['epoches']
""" if multi_gpu:
    batch_size = batch_size * len(gpu_idx)
    learning_rate = learning_rate * len(gpu_idx) """

logger_base.info("batch_size= {}".format(batch_size))
logger_base.info("learning_rate= {}".format(learning_rate))
logger_base.info("epoches= {}".format(epoches))


# 训练终端打印等参数
val_freq = paramDict['val_freq']
print_freq = paramDict['print_freq']
save_freq = paramDict['save_freq']


# 模型
model = paramDict['model']
if paramDict['resume_training']:
    model.load_state_dict( torch.load( os.path.join(checkpointDir, paramDict['checkpointName']) ), strict=False)
""" if multi_gpu:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpu_idx) """
model = model.cuda() # 模型放GPU上；
logger_base.info("modelinfo:" + model.getname())

# 数据集
cars_train_annos_Path = os.path.join(datasetDir, 'cars_train_annos.mat')
img_dir = os.path.join(datasetDir, 'cars_train')
data_train = read_annos_to_np(cars_train_annos_Path)
dataset = paramDict['dataset']( img_dir, 
                                data_train, 
                                aug=paramDict['augment'], 
                                img_size = paramDict['img_size'],
                                img_mode = paramDict['img_mode'])
    
logger_base.info("augment:" + str(paramDict['augment']))
logger_base.info("img_size:" + str(paramDict['img_size']))
logger_base.info("img_mode:" + str(paramDict['img_mode']))

# 切分训练和验证
lenTrain = int(len(dataset)*0.8)
lenValid = len(dataset)-lenTrain
train_dataset, valid_dataset = torchData.random_split(dataset, [lenTrain, lenValid])

train_sampler = None
""" if multi_gpu:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) """

trainLoader= torchData.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, sampler=train_sampler, pin_memory=True)
validLoader= torchData.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 训练
best_acc = 0.5
loss_fn = paramDict['loss'] 

# 只优化除backbone之外的参数
if paramDict['ignore_optim_flag']:
    for k, v in model.named_parameters():
        #v.requires_grad = True
        if k.find(paramDict['ignore_backbone_name']) >= 0:
            v.requires_grad = False
        else:
            #v.data.zero_()
            #optim_params.append(v)
            logger_base.info(
                'Params [{:s}] will optimize.'.format(k))

#优化器
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate, 
                            betas=[0.9, 0.999], 
                            weight_decay=0)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3)
logger_base.info("optimier:" + str(optimizer))
logger_base.info("ignore_optim_flag:" + str(paramDict['ignore_optim_flag']))
if paramDict['ignore_optim_flag']:
    logger_base.info("ignore part:" + paramDict['ignore_backbone_name'])

for epoch in range(epoches):
    model.train()
    totalLoss = 0
    for step, data in enumerate(trainLoader, start=0):
        imgs, labels = data
        
        optimizer.zero_grad()
        labels_predict = model(imgs)
        loss = loss_fn(labels_predict, labels)
        loss.backward()
        optimizer.step()
        
        rate = (step + 1) / len(trainLoader)
        totalLoss += loss
        print("\r epoch:%s %3.0f%% | train loss: %.5f | totalLoss = %.5f" % (epoch, int(rate * 100), loss, totalLoss/(step+1)), end="  ")
    
    logger_base.info("EPOCH: {} | batch_avg_Loss: {:.5f}".format(epoch, totalLoss / len(trainLoader)))    

    # 验证
    if epoches % val_freq == 0:
        model.eval()
        with torch.no_grad():
            label_all = []
            label_predict_all = []
            for data in validLoader:
                imgs, labels = data
                labels_predict = model(imgs)
                
                labels = labels.cpu().numpy()
                labels_predict = np.argmax(labels_predict.cpu().numpy(), axis = 1)
                for i in range(labels.shape[0]):
                    label_all.append(labels[i])
                    label_predict_all.append(labels_predict[i])
            label_all = np.array(label_all)
            label_predict_all = np.array(label_predict_all)
            
            acc = sum(label_all == label_predict_all) / len(label_all)
            logger_val.info('Validation Acc= {}'.format(acc))
    
            if acc > best_acc:
                best_acc = acc
                checkpointName = "checkpoint_epoch" + str(epoch) +'_acc'+str(acc)+ '.pth'
                print("Saving checkpoint:", checkpointName)
                torch.save(model.state_dict(), os.path.join(checkpointDir ,checkpointName))
                logger_val.info("Saving checkpoint:" + checkpointName)
