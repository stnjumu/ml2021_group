from ModelLib.swin import SwinNet
from ModelLib.efficientnet_v2 import *
from UtilLib.Read_annos_mat import read_annos_to_np
from ModelLib.Model1 import Model1
from ModelLib.resnet101 import RenNet101_head
from ModelLib.resnet import resnet50
from DatasetLib.DatasetTorch import DatasetTorch
from Submission.client import submitCarDemand_evaluate

import UtilLib.logger as Logger
import torch
import torch.utils.data as torchData
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from tqdm import tqdm
import time
from datetime import datetime
import timm

# 参数字典
paramDict = {
    'batch_size': 2, 
    'learning_rate': 1e-3,
    'epoches': 100, 
    'loss': torch.nn.CrossEntropyLoss(),

    # 模型设置
    'model': effnetv2_m(num_classes=196), # 自定义模型
    'resume_training': False, # 继续训练
    'checkpointName': 'swin_large_patch4_window12_384_22kto1k.pth', # 检查点名称
    'ignore_optim_flag': False, # 忽略部分预训练模型参数
    'ignore_backbone_name': 'backbone', # 要忽略的预训练参数名称
    
    # 数据集设置
    'DatasetClass': DatasetTorch, # 自定义数据库

     # 日志设置
    'datasetDir': './dataset/', # 数据集存放路径
    'checkpointDir':  './checkpoint/', # 检查点路径
    'submissionDir':  './submission/', # 检查点路径
    'log_path': 'log', # 日志路径
    'val_freq' : 1, # 每隔epoch验证
    'test_freq' : 1, # 每隔epoch测试，最好和val_freq一样
    'print_freq' : 100, # 每隔step计算准确率
    'save_freq' : 10, # 每隔epoch存储模型，暂未使用
    
    # 提交账号，为避免同时提交的冲突，先使用个人账号密码
    'studentID': '',        # str类型
    'password':''           # str类型

}

# 一些配置
torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(0)
datasetDir = paramDict['datasetDir'] # 数据集文件夹
checkpointDir = paramDict['checkpointDir'] # 创建保存checkpoint的文件夹
submissionDir = paramDict['submissionDir'] # 创建保存checkpoint的文件夹
os.makedirs(checkpointDir, exist_ok=True)

# 日志配置
log_path = paramDict['log_path']
Logger.setup_logger(None,log_path,
                    '{}_train'.format(datetime.now().strftime('%y%m%d_%H%M')), level=logging.INFO, screen=True)
Logger.setup_logger('val',log_path, '{}_val'.format(datetime.now().strftime('%y%m%d_%H%M')), level=logging.INFO)
logger_base = logging.getLogger('base')
logger_val = logging.getLogger('val')
for k,v in paramDict.items():
    logger_base.info("{} = {}".format(k, v))

# 超参数
batch_size = paramDict['batch_size']
learning_rate = paramDict['learning_rate']
epoches = paramDict['epoches']
logger_base.info("batch_size= {}".format(batch_size))
logger_base.info("learning_rate= {}".format(learning_rate))
logger_base.info("epoches= {}".format(epoches))

# 训练参数
val_freq = paramDict['val_freq']
test_freq = paramDict['test_freq']
print_freq = paramDict['print_freq']
save_freq = paramDict['save_freq']


# 模型
model = paramDict['model']
if paramDict['resume_training']:
    checkpoint = torch.load(os.path.join(checkpointDir , paramDict['checkpointName']), map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    logger_base.info(
            'Load model from {}/{}.'.format(checkpointDir, paramDict['checkpointName']))
model = model.cuda() # 模型放GPU上；

# 训练数据集
cars_train_annos_Path = os.path.join(datasetDir, 'cars_train_annos.mat')
img_dir = os.path.join(datasetDir, 'cars_train')
data_all = read_annos_to_np(cars_train_annos_Path)
# 切分训练和验证
np.random.shuffle(data_all) # 随机排序, 默认axis=0
lenTrain = int(data_all.shape[0]*0.8) # 训练集比例
data_train = data_all[:lenTrain, :]
data_test = data_all[lenTrain:, :]
dataset_train = paramDict['DatasetClass'](img_dir, data_train)
dataset_test = paramDict['DatasetClass'](img_dir, data_test, split='valid')

trainLoader= torchData.DataLoader(dataset_train,batch_size=batch_size,shuffle=True,drop_last=True)
validLoader= torchData.DataLoader(dataset_test,batch_size=batch_size,shuffle=True,drop_last=True)

# 测试数据集
cars_test_annos_Path = os.path.join(datasetDir, 'cars_test_annos.mat')
img_dir = os.path.join(datasetDir, 'cars_test')
data_test = read_annos_to_np(cars_test_annos_Path)
dataset = paramDict['dataset'](img_dir, data_test, split='test') 

testLoader = torchData.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

# 训练
# best_acc = 0.50
best_score = 0.30
loss_fn = paramDict['loss']

# 只优化除backbone之外的参数
if paramDict['ignore_optim_flag']:
    for k, v in model.named_parameters():
        if k.find(paramDict['ignore_backbone_name']) >= 0:
            v.requires_grad = False
        else:
            logger_base.info(
                'Params [{:s}] will optimize.'.format(k))
#优化器
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                             lr=learning_rate, betas=[0.9, 0.999], weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-3)
# ? Adam优化器会自动调整学习率，可能不需要scheduler了

logger_base.info("optimier:" + str(optimizer))
logger_base.info("ignore_optim_flag:" + str(paramDict['ignore_optim_flag']))
if paramDict['ignore_optim_flag']:
    logger_base.info("ignore part:" + paramDict['ignore_backbone_name'])


for epoch in range(epoches):
    
    model.train()
    totalLoss = 0
    for step, data in enumerate(trainLoader, start=0):
        imgs, labels = data
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        labels_predict = model(imgs)
        loss = loss_fn(labels_predict, labels)
        loss.backward()
        optimizer.step()
        
        rate = (step + 1) / len(trainLoader)
        totalLoss += loss

        if step % print_freq == 0:
            print("\repoch:%s train loss:%3.0f%%:%.4f, totalLoss = %.4f" % (epoch, int(rate * 100), loss, totalLoss/(step+1)))    
            acc = (labels_predict.argmax(dim=1) == labels).float().mean().item()
            logger_base.info("step:%s , accuracy = %.4f" % (step, acc))
        else:
            print("\repoch:%s train loss:%3.0f%%:%.4f, totalLoss = %.4f" % (epoch, int(rate * 100), loss, totalLoss/(step+1)), end="  ")
    # 验证
    if epoches % val_freq == 0:
        model.eval()
        with torch.no_grad():
            label_all = []
            label_predict_all = []
            for data in validLoader:
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()
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
    
            # if acc > best_acc:
            #     best_acc = acc
            #     checkpointName = "checkpoint_epoch" + str(epoch) +'_acc'+str(acc)+ '.pth'
            #     print("Saving checkpoint:", checkpointName)
            #     torch.save(model.state_dict(), os.path.join(checkpointDir ,checkpointName))
            
    if epoches % test_freq == 0:
        model.eval()
        with torch.no_grad():
            # 预测
            output_name = datetime.now().strftime('%y%m%d_%H%M%S')
            submissionPath = 'Submission/submission/{}_submission.txt'.format(output_name)
            label_all = []
            label_predict_all = []
            pbar = tqdm(testLoader)
            start = time.time()
            for data in pbar:
                imgs, file_name = data
                imgs= imgs.cuda()
                labels_predict = model(imgs)
                labels_predict = np.argmax(labels_predict.cpu().numpy(), axis = 1)
                with open(submissionPath, 'a+') as f:
                    for i in range(labels_predict.shape[0]):
                        print(file_name[i], labels_predict[i]+1, file=f)
            logger_test.info('Finished, Time cost is {}.'.format(time.time()-start))
            
            # 提交
            score = submitCarDemand_evaluate(submissionPath, paramDict['studentID'], paramDict['password'])
            with open('Submission/All_Submission.txt', 'a+') as f:
                print(output_name, '\t', score, file=f)
                
            # 保存模型
            if score > best_score:
                best_score = score
                checkpointName = "checkpoint_epoch" + str(epoch) +'_score'+str(score)+ '.pth'
                print("Saving checkpoint:", checkpointName)
                torch.save(model.state_dict(), os.path.join(checkpointDir ,checkpointName))