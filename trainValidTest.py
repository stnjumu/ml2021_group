from ModelLib.swin import SwinNet
from ModelLib.efficientnet_v2_pretrain import *
from ModelLib.tresnet_v2 import TResnetL_V2
from UtilLib.Read_annos_mat import read_annos_to_np
from ModelLib.Model1 import Model1
from ModelLib.resnet101 import RenNet101_head
from ModelLib.resnet import resnet50
from DatasetLib.DatasetTorch import DatasetTorch
import UtilLib.logger as Logger
from UtilLib.loss import FocalLoss, SoftCrossEntropyLoss

import UtilLib.logger as Logger
import torch
import torch.utils.data as torchData
import matplotlib.pyplot as plt
import numpy as np
import logging
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import time
from datetime import datetime
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="4"
torch.set_default_tensor_type('torch.FloatTensor')

# 参数字典
paramDict = {
    'verbose': False, # 调试
    'valid_enable': False, # 是否划分一部分数据集为验证集
    'test_enable': True, # 是否测试，测试默认提交到leaderboard
    'tb_logger': True, # 是否开启tensor logger
    'batch_size': 12, 
    'learning_rate': 1e-4,
    'lr_scheduler_enable' : True,
    'epoches': 50, 
    # 'loss': torch.nn.CrossEntropyLoss(),
    'loss': FocalLoss(),

    # 模型设置
    'model': EffNetV2(), # 自定义模型
    'resume_training': False, # 继续训练
    'checkpointPath': './checkpoint/stanford_cars_tresnet-l-v2_96_27.pth', # 检查点名称
    'ignore_optim_flag': False, # 忽略部分预训练模型参数
    'ignore_backbone_name': 'backbone', # 要忽略的预训练参数名称
    
    # 数据集设置
    'DatasetClass': DatasetTorch, # 自定义数据库

     # 日志设置
    'datasetDir': './dataset/', # 数据集存放路径
    'logPath': 'log', # 日志路径
    'val_freq' : 2, # 每隔epoch验证
    'print_freq' : 100, # 每隔step计算准确率
    'save_freq' : 1, # 每隔epoch存储模型
}

class Trainer():
    def __init__(self):
        self.runnerInit()
        self.logInit()
        self.datasetInit()
        self.modelInit()

    def runnerInit(self):
        self.verbose = paramDict['verbose']
        if self.verbose:    
            paramDict['valid_enable'] = True
            paramDict['val_freq'] = 1
            paramDict['print_freq'] = 5
            paramDict['save_freq'] = 1
            paramDict['tb_logger'] = True
            
        self.valid_enable = paramDict['valid_enable']
        self.test_enable = paramDict['test_enable']
        self.epoches = paramDict['epoches']
        self.total_train_step = 0
        self.total_valid_step = 0
        self.best_acc = 0.8

        
    def datasetInit(self):
        # 数据集
        self.batch_size = paramDict['batch_size']

        datasetDir = paramDict['datasetDir'] # 数据集文件夹
        cars_train_annos_Path = os.path.join(datasetDir, 'cars_train_annos.mat')
        img_dir = os.path.join(datasetDir, 'cars_train')
        data_all = read_annos_to_np(cars_train_annos_Path)

        # 切分训练和验证
        train_val_ratio = 1
        if self.valid_enable:
            train_val_ratio = 0.8
        lenTrain = int(data_all.shape[0]*train_val_ratio)
        data_train = data_all[:lenTrain, :]
        data_val = data_all[lenTrain:, :]
        if self.verbose:
            data_train = data_all[:self.batch_size*2, :]
            data_val = data_all[:self.batch_size*2, :]
        dataset_train = paramDict['DatasetClass'](img_dir, data_train, split='train')
        dataset_val = paramDict['DatasetClass'](img_dir, data_val, split='val')

        self.trainLoader= torchData.DataLoader(dataset_train,batch_size=self.batch_size,shuffle=True,drop_last=True)
        if self.valid_enable:
            self.validLoader= torchData.DataLoader(dataset_val,batch_size=self.batch_size,shuffle=False,drop_last=True)    

    def logInit(self):
        # 日志配置
        self.logPath = "{}/Train_{}".format(paramDict['logPath'], datetime.now().strftime('%y%m%d_%H%M'))
        self.resultDir = "{}/result".format(self.logPath)
        os.makedirs(self.logPath, exist_ok=True)
        os.makedirs(self.resultDir, exist_ok=True)

        self.tb_logger_enable = paramDict['tb_logger']
        if self.tb_logger_enable:
            self.tb_logger = SummaryWriter(log_dir=self.logPath)
        Logger.setup_logger(None,self.logPath,'train', level=logging.INFO, screen=True)
        Logger.setup_logger('val',self.logPath,'val', level=logging.INFO)
        Logger.setup_logger('test',self.logPath,'test', level=logging.INFO, screen=True)
        self.logger_base = logging.getLogger('base')
        self.logger_val = logging.getLogger('val')
        self.logger_test = logging.getLogger('test')
        for k,v in paramDict.items():
            self.logger_base.info("{} = {}".format(k, v))
        
        # 打印频率
        self.val_freq = paramDict['val_freq']
        self.print_freq = paramDict['print_freq']
        self.save_freq = paramDict['save_freq']

        # 检查点存放
        self.checkpointSaveDir = "{}/checkpoint".format(self.logPath)
        os.makedirs(self.checkpointSaveDir, exist_ok=True)

    def modelInit(self):
        # 模型
        model = paramDict['model']
        if paramDict['resume_training']:
            checkpoint = torch.load(os.path.join(paramDict['checkpointPath']), map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            self.logger_base.info(
                    'Load model from {}/{}.'.format(paramDict['checkpointPath']))
        self.model = model.cuda() # 模型放GPU上；

        # 训练
        self.loss_fn = paramDict['loss'] 
        self.lr = paramDict['learning_rate']

        # 只优化除backbone之外的参数
        if paramDict['ignore_optim_flag']:
            optim_params = []
            for k, v in model.named_parameters():
                v.requires_grad = True
                if k.find(paramDict['ignore_backbone_name']) >= 0:
                    v.requires_grad = False
                else:
                    optim_params.append(v)
                    self.logger_base.info('Params [{:s}] will optimize.'.format(k))
            self.optimizer = torch.optim.Adam(params=optim_params, lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        if paramDict['lr_scheduler_enable']:
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,base_lr=1e-6, max_lr=1e-4, gamma=0.99994, cycle_momentum=False)
        
    def _train(self, epoch):
        self.model.train()
        totalLoss = 0
        for step, data in enumerate(self.trainLoader, start=0):
            self.total_train_step += 1
            imgs, labels = data
            imgs, labels = imgs.cuda(), labels.cuda()
            self.optimizer.zero_grad()
            labels_predict = self.model(imgs)
            loss = self.loss_fn(labels_predict, labels)
            loss.backward()
            self.optimizer.step()
            if paramDict['lr_scheduler_enable']:
                self.lr_scheduler.step()
            rate = (step + 1) / len(self.trainLoader)
            totalLoss += loss

            if step % self.print_freq == 0:
                print("\repoch:%s train loss:%3.0f%%:%.4f, totalLoss = %.4f" % (epoch, int(rate * 100), loss, totalLoss/(step+1)))    
                acc = (labels_predict.argmax(dim=1) == labels).float().mean().item()
                self.logger_base.info("step:%s , accuracy = %.4f, loss = %.4f" % (step, acc, totalLoss/(step+1)))
                if self.tb_logger_enable:
                    self.tb_logger.add_scalar('Train_Acc', acc, self.total_train_step//self.print_freq)
                    self.tb_logger.add_scalar('Train_Loss', totalLoss/(step+1), self.total_train_step//self.print_freq)
                    self.tb_logger.add_image('Train_Img', imgs[0], self.total_train_step//self.print_freq)
                    self.tb_logger.add_scalar('Lr', self.optimizer.param_groups[0]["lr"], self.total_train_step//self.print_freq)
            else:
                print("\repoch:%s train loss:%3.0f%%:%.4f, totalLoss = %.4f" % (epoch, int(rate * 100), loss, totalLoss/(step+1)), end="  ")
        if epoch % self.save_freq==0:
            checkpointName = "checkpoint_epoch" + str(epoch) +'_freq'+ '.pth'
            print("Saving checkpoint:", checkpointName)
            torch.save(self.model.state_dict(), os.path.join(self.checkpointSaveDir ,checkpointName))
            self._test(checkpointPath = os.path.join(self.checkpointSaveDir ,checkpointName), epoch=epoch)

    def _val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            label_all = []
            label_predict_all = []
            pbar = tqdm(self.validLoader)
            for data in pbar:
                self.total_valid_step += 1
                imgs, labels = data
                imgs, labels = imgs.cuda(), labels.cuda()
                labels_predict = self.model(imgs)
                
                labels = labels.cpu().numpy()
                labels_predict = np.argmax(labels_predict.cpu().numpy(), axis = 1)
                for i in range(labels.shape[0]):
                    label_all.append(labels[i])
                    label_predict_all.append(labels_predict[i])

                if self.tb_logger_enable:
                    self.tb_logger.add_image('Validation_Img', imgs[0], self.total_valid_step)
            label_all = np.array(label_all)
            label_predict_all = np.array(label_predict_all)
            
            acc = sum(label_all == label_predict_all)/len(label_all)
            if self.tb_logger_enable:
                self.tb_logger.add_scalar('Validation_Acc', acc, epoch)
            self.logger_val.info('Validation Acc= {}'.format(acc))

            if acc > self.best_acc:
                self.best_acc = acc
                checkpointName = "checkpoint_epoch" + str(epoch) +'_acc'+str(acc)+ '.pth'
                print("Saving checkpoint:", checkpointName)
                torch.save(self.model.state_dict(), os.path.join(self.checkpointSaveDir ,checkpointName))
                self._test(checkpointPath = os.path.join(self.checkpointSaveDir ,checkpointName), epoch=epoch)

    def _test(self, checkpointPath, epoch, gpu_id=1):
        if self.test_enable:
            try:
                self.logger_test.info('Epoch {} : Test and sumbit to the leaderboard'.format(epoch))
                os.popen('python testSubmit.py -gpu {} -m {} -p {} >> log/ranking.log'.format(gpu_id, checkpointPath, self.resultDir))

            except (BrokenPipeError, IOError):
                self.logger_test.info('Epoch {} : Test Error, Checkpoint is {}'.format(IOError, checkpointPath))
    def run(self):
        for epoch in range(1, self.epoches+1):
            self._train(epoch)
            if self.valid_enable and epoch % self.val_freq == 0:
                self._val(epoch)
            
if __name__ == "__main__":
    worker = Trainer()
    worker.run()
