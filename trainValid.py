from UtilLib.Read_annos_mat import read_annos_to_np
from ModelLib.Model1 import Model1
from ModelLib.resnet101 import RenNet101_head
from ModelLib.resnet import resnet50
from DatasetLib.Dataset1 import Dataset1

import torch
import torch.utils.data as torchData
import matplotlib.pyplot as plt
import numpy as np
import os

# 一些配置
torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.set_device(0)
datasetDir = './dataset/' # 数据集文件夹
checkpointDir = './checkpoint/' # 创建保存checkpoint的文件夹
if not os.path.exists(checkpointDir):
    os.makedirs(checkpointDir)

# 超参数
batch_size = 15
learning_rate = 0.001
epoches = 20
print("batch_size= ", batch_size)
print("learning_rate= ", learning_rate)
print("epoches= ", epoches)


# 模型
model = resnet50()
#model.load_state_dict( torch.load( os.path.join(checkpointDir ,'checkpoint_epoch10.pth') )['net_state_dict'], strict=False)
model.load_state_dict( torch.load( os.path.join(checkpointDir ,'checkpoint_epoch10.pth') ), strict=False)
model = model.cuda() # 模型放GPU上；

# 数据集
cars_train_annos_Path = os.path.join(datasetDir, 'cars_train_annos.mat')
img_dir = os.path.join(datasetDir, 'cars_train')
data_train = read_annos_to_np(cars_train_annos_Path)
dataset = Dataset1(img_dir, data_train)
# 切分训练和验证
lenTrain = int(len(dataset)*0.8)
lenValid = len(dataset)-lenTrain
train_dataset, valid_dataset = torchData.random_split(dataset, [lenTrain, lenValid])

# 训练
best_acc = 0.04
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.CrossEntropyLoss()
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
        print("\repoch:%s iteration loss:%3.0f%%:%.4f, totalLoss = %.4f" % (epoch, int(rate * 100), loss, totalLoss/(step+1)), end="  ")
    
    # 验证
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
        print('acc= ', acc)
        if acc > best_acc:
            best_acc = acc
            # for img in imgs:
            #     img=img.cpu().numpy()
            #     img = img.transpose([1,2,0]) # 3 * w * h -> w * h * 3
            #     plt.figure()
            #     plt.imshow(img)
            #     plt.show()
            # break
    
            # 保存模型
            checkpointName = "checkpoint_epoch" + str(epoch) +'_acc'+str(acc)+ '.pth'
            print("Saving checkpoint:", checkpointName)
            torch.save(model.state_dict(), os.path.join(checkpointDir ,checkpointName))