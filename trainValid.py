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
import cv2
import logging
import os
from datetime import datetime

# 参数字典
paramDict = {
     # 训练设置
    'batch_size': 10, 
    'learning_rate': 1e-3,
    'epoches': 100, 
    'loss': torch.nn.CrossEntropyLoss(),

    # 模型设置
    'model': RenNet101_head(), # 自定义模型
    'resume_training': False, # 继续训练
    'checkpointName': 'checkpoint_epoch10.pth', # 检查点名称
    'ignore_optim_flag': False, # 忽略部分预训练模型参数
    'ignore_backbone_name': 'base_backbone', # 要忽略的预训练参数名称
    
    # 数据集设置
    'dataset': DatasetTorch, # 自定义数据库
    'augment': False, # 训练集数据增强,
    'img_size': [384,384], # 图片缩放至尺寸

     # 日志设置
    'datasetDir': './dataset/', # 数据集存放路径
    'checkpointDir':  './checkpoint/', # 检查点路径
    'log_path': 'log', # 日志路径
    'val_freq' : 2, # 每隔epoch验证
    'print_freq' : 10, # 每隔step计算准确率
    'save_freq' : 10, # 每隔epoch存储模型，暂未使用
}

# 一些配置
torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(0)
datasetDir = paramDict['datasetDir'] # 数据集文件夹
checkpointDir = paramDict['checkpointDir'] # 创建保存checkpoint的文件夹
os.makedirs(checkpointDir, exist_ok=True)

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
logger_base.info("batch_size= {}".format(batch_size))
logger_base.info("learning_rate= {}".format(learning_rate))
logger_base.info("epoches= {}".format(epoches))

# 训练参数
val_freq = paramDict['val_freq']
print_freq = paramDict['print_freq']
save_freq = paramDict['save_freq']


# 模型
model = paramDict['model']
if paramDict['resume_training']:
    model.load_state_dict( torch.load( os.path.join(checkpointDir , paramDict['checkpointName']) ), strict=False)
model = model.cuda() # 模型放GPU上；

# 数据集
cars_train_annos_Path = os.path.join(datasetDir, 'cars_train_annos.mat')
img_dir = os.path.join(datasetDir, 'cars_train')
data_train = read_annos_to_np(cars_train_annos_Path)
dataset = paramDict['dataset'](img_dir, data_train, aug=paramDict['augment']) 
# 切分训练和验证
lenTrain = int(len(dataset)*0.8)
lenValid = len(dataset)-lenTrain
train_dataset, valid_dataset = torchData.random_split(dataset, [lenTrain, lenValid])

# 训练
best_acc = 0.04
# loss_fn = torch.nn.MSELoss()
loss_fn = paramDict['loss'] 
# 只优化除backbone之外的参数
if paramDict['ignore_optim_flag']:
    optim_params = []
    for k, v in model.named_parameters():
        v.requires_grad = True
        if k.find(paramDict['ignore_backbone_name']) >= 0:
            v.requires_grad = False
        else:
            v.data.zero_()
            optim_params.append(v)
            logger_base.info(
                'Params [{:s}] initialized to 0 and will optimize.'.format(k))
    optimizer = torch.optim.Adam(params=optim_params, lr=learning_rate)
else:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

for epoch in range(epoches):
    
    trainLoader= torchData.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    validLoader= torchData.DataLoader(valid_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    
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
        print("\repoch:%s train loss:%3.0f%%:%.4f, totalLoss = %.4f" % (epoch, int(rate * 100), loss, totalLoss/(step+1)), end="  ")

        if step % print_freq == 0:
            acc = (labels_predict.argmax(dim=1) == labels).float().sum().item()
            logger_base.info("step:%s , accuracy = %.4f" % (epoch, acc))

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
            
            acc = sum(label_all == label_predict_all)/len(label_all)
            logger_val.info('Validation Acc= {}'.format(acc))
                # for img in imgs:
                #     img=img.cpu().numpy()
                #     img = img.transpose([1,2,0]) # 3 * w * h -> w * h * 3
                #     plt.figure()
                #     plt.imshow(img)
                #     plt.show()
                # break
    
            if acc > best_acc:
                best_acc = acc
                checkpointName = "checkpoint_epoch" + str(epoch) +'_acc'+str(acc)+ '.pth'
                print("Saving checkpoint:", checkpointName)
                torch.save(model.state_dict(), os.path.join(checkpointDir ,checkpointName))
