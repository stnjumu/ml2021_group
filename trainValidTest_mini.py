from ModelLib.swin import SwinNet
from ModelLib.efficientnet_v2_pretrain import *
from UtilLib.Read_annos_mat import read_annos_to_np
from ModelLib.Model1 import Model1
from ModelLib.resnet101 import RenNet101_head
from ModelLib.resnet import resnet50
from DatasetLib.DatasetTorch import DatasetTorch
import UtilLib.logger as Logger
from UtilLib.loss import FocalLoss, SoftCrossEntropyLoss

import torch
import torch.utils.data as torchData
import matplotlib.pyplot as plt
import numpy as np
import logging
#from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
import time
from datetime import datetime
import timm
import requests

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.set_default_tensor_type('torch.FloatTensor')

# 参数字典
paramDict = {
    'discription': 'mpp3: resnest101e只用一张卡，训练0~100', # 一句话描述此次训练
    'verbose': False, # 调试
    'valid_enable': True, # 是否划分一部分数据集为验证集
    'valid_epoches': 20, # 前valid_epoches个epoch验证不提交，后面的epoch提交不验证；
    'train_val_ratio': 0.8, # 划分训练/验证集的话，训练集所占比例；
    'test_enable': True, # 是否测试，测试默认提交到leaderboard
    'tb_logger': False, # 是否开启tensor logger
    'batch_size': 30, 
    'learning_rate': 1e-4,
    'lr_scheduler_enable' : True,
    'epoches': 100, 
    # 'loss': torch.nn.CrossEntropyLoss(),
    'loss': FocalLoss(),
    # 'loss': SoftCrossEntropyLoss(),

    # 模型设置
    # 'model': timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=196), # 自定义模型
    'model': timm.create_model('resnest101e', pretrained= True, num_classes = 196), # 自定义模型
    # 'model': timm.create_model('resnest269e', pretrained= True, num_classes = 196), # 自定义模型
    'resume_training': False, # 继续训练
    'checkpointPath': './log/Train_211229_0150/checkpoint/checkpoint_epoch49_scoreNone.pth', # 检查点名称
    'ignore_optim_flag': False, # 忽略部分预训练模型参数
    'ignore_backbone_name': 'backbone', # 要忽略的预训练参数名称
    
    # 数据集设置
    'DatasetClass': DatasetTorch, # 自定义数据库
    # 'DatasetClass': DatasetTorch416, # 自定义数据库

     # 日志设置
    'datasetDir': '/home/zhounan_2021/code/MLfinalwork/ml2021_group/dataset', # 数据集存放路径
    'logPath': 'log', # 日志路径
    'val_freq' : 1, # 每隔epoch验证
    'print_freq' : 100, # 每隔step计算准确率
    'save_freq' : 1, # 每隔epoch存储模型
    
    # 提交设置
    'studentID': 'ZY2106345',
    'password': 'gagaga'
}

class Trainer():
    def __init__(self):
        self.runnerInit()
        self.logInit()
        self.datasetInit()
        self.modelInit()

    def runnerInit(self):
        self.verbose = paramDict['verbose']
            
        self.valid_enable = paramDict['valid_enable']
        self.valid_epoches = paramDict['valid_epoches']
        self.test_enable = paramDict['test_enable']
        self.epoches = paramDict['epoches']
        self.total_train_step = 0
        self.total_valid_step = 0
        # self.best_acc = 0.9 # 不再使用和更新best_acc，保存所有大于下界的模型。
        self.acc_lower_bound = 0.92
        self.score_lower_bound = 0.93

    def datasetInit(self):
        # 测试数据集
        self.batch_size = paramDict['batch_size']

        datasetDir = paramDict['datasetDir'] # 数据集文件夹
        cars_train_annos_Path = os.path.join(datasetDir, 'cars_train_annos.mat')
        train_img_dir = os.path.join(datasetDir, 'cars_train')
        data_all = read_annos_to_np(cars_train_annos_Path)
        # 切分训练和验证
        np.random.shuffle(data_all) # 随机排序
        train_val_ratio = paramDict['train_val_ratio']
        lenTrain = int(data_all.shape[0]*train_val_ratio)
        data_train = data_all[:lenTrain, :]
        data_val = data_all[lenTrain:, :]
        if self.verbose:
            data_train = data_all[:self.batch_size*2, :]
            data_val = data_all[:self.batch_size*2, :]
            data_all = data_all[:self.batch_size*2, :]
        dataset_all = paramDict['DatasetClass'](train_img_dir, data_all, split='train')
        dataset_train = paramDict['DatasetClass'](train_img_dir, data_train, split='train')
        dataset_val = paramDict['DatasetClass'](train_img_dir, data_val, split='val')
        
        self.dataset_all = dataset_all

        self.trainLoader= torchData.DataLoader(dataset_train,batch_size=self.batch_size,shuffle=True,drop_last=True)
        if self.valid_enable:
            self.validLoader= torchData.DataLoader(dataset_val,batch_size=self.batch_size,shuffle=False,drop_last=True)    

        # 测试数据集
        cars_test_annos_Path = os.path.join(datasetDir, 'cars_test_annos.mat')
        test_img_dir = os.path.join(datasetDir, 'cars_test')
        data_test = read_annos_to_np(cars_test_annos_Path)
        dataset_test = paramDict['DatasetClass'](test_img_dir, data_test, split='test') 
        self.testLoader = torchData.DataLoader(dataset_test, batch_size = self.batch_size, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
        
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
        Logger.setup_logger('val_or_test',self.logPath,'val_or_test', level=logging.INFO, screen=False) # BUG: screen = True会在控制台打印2次
        self.logger_base = logging.getLogger('base')
        self.logger_base.info(paramDict['discription'])
        self.logger_val_or_test = logging.getLogger('val_or_test')
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
            model.load_state_dict(checkpoint, strict=False)
            self.logger_base.info(
                    'Load model from {}.'.format(paramDict['checkpointPath']))
        self.model = model.cuda()

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
            
            print("\repoch:%s train loss:%3.0f%%:%.4f, totalLoss = %.4f" % (epoch, int(rate * 100), loss, totalLoss/(step+1)), end="  ")
            
            if step % self.print_freq == 0 and self.tb_logger_enable:
                # self.tb_logger.add_scalar('Train_Acc', acc, self.total_train_step//self.print_freq)
                self.tb_logger.add_scalar('Train_Loss', totalLoss/(step+1), self.total_train_step//self.print_freq)
                self.tb_logger.add_image('Train_Img', imgs[0], self.total_train_step//self.print_freq)
                self.tb_logger.add_scalar('Lr', self.optimizer.param_groups[0]["lr"], self.total_train_step//self.print_freq)

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
            self.logger_base.info('Validation Acc= {}'.format(acc))

            if acc > self.acc_lower_bound:
                checkpointName = "checkpoint_epoch" + str(epoch) +'_acc'+str(acc)+ '.pth'
                self.logger_base.info("Saving checkpoint: {}".format(checkpointName))
                torch.save(self.model.state_dict(), os.path.join(self.checkpointSaveDir ,checkpointName))

    def _test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.testLoader)
            for data in pbar:
                imgs, file_name = data
                imgs= imgs.cuda()
                labels_predict = self.model(imgs)
                labels_predict = np.argmax(labels_predict.cpu().numpy(), axis = 1)
                submissionPath = '{}/submission_epoch{}.txt'.format(self.resultDir, epoch)
                with open(submissionPath, 'a+') as f:
                    for i in range(labels_predict.shape[0]):
                        print(file_name[i], labels_predict[i]+1, file=f)
            score = self._submit(epoch, submissionPath)
            if score == None or float(score) > self.score_lower_bound: # None出错则直接保存，否则判断score
                checkpointName = "checkpoint_epoch" + str(epoch) +'_score'+str(score)+ '.pth'
                self.logger_base.info("Saving checkpoint: {}".format(checkpointName))
                torch.save(self.model.state_dict(), os.path.join(self.checkpointSaveDir ,checkpointName))
               
    
    def _submit(self, epoch, submissionPath): 
        if not self.test_enable: # 不提交
            return None 
        
        # 在_test()中用到
        self.logger_val_or_test.info('Epoch {} : Test and sumbit to the leaderboard'.format(epoch))

        problem = "FineGrainedCar_evaluate"
        ip = "115.236.52.125"
        port = "4000"
        sid = paramDict['studentID']
        token = paramDict['password']
        with open(submissionPath) as f:
            d = list(f.readlines())
        
        try:
            score = self.__submit(ip, port, sid, token, d, problem) 
            self.logger_val_or_test.info('Epoch {} : Result:[{}], File:[{}]'.format(epoch, score, submissionPath))
            return score
        except(BrokenPipeError, IOError):
            self.logger_val_or_test.info('Epoch {} : Test Error, Checkpoint is {}'.format(epoch, IOError))
            return None
        # print('Result: ', score, 'Submission Txt Path:[', logPath, 'CheckPoint Path:', checkpointPath)
    
    def __submit(self, ip, port, sid, token, ans, problem):
        print("正在提交...", end=' ')
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

        if "auth_error" in response:
            print("您的认证信息有误")
            return response["auth_error"]
        elif "error" not in response:
            print("测试完成，请查看分数")
            return response["result"]
        else:
            print("提交文件存在问题，请查看error信息")
            return response["error"]["data"]["message"]
    
    def run(self):
        for epoch in range(1, self.epoches+1):
            if self.valid_enable:  # 如果选择验证
                if epoch <= self.valid_epoches: # 仅验证
                    self._train(epoch)
                    self._val(epoch)
                else: # 仅测试
                    if epoch == self.valid_epoches + 1:
                        self.trainLoader = torchData.DataLoader(self.dataset_all ,batch_size=self.batch_size,shuffle=True,drop_last=True)
                    self._train(epoch)
                    self._test(epoch)
            else: # 没选择验证，仅测试
                self.trainLoader = torchData.DataLoader(self.dataset_all ,batch_size=self.batch_size,shuffle=True,drop_last=True)
                self._train(epoch)
                self._test(epoch)


if __name__ == "__main__":
    worker = Trainer()
    worker.run()
