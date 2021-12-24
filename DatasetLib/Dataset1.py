# 数据集示例
# 只是读取图片，裁剪后resize到256*256
import torch
import torch.utils.data as torchData
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

class Dataset1(torchData.Dataset):
    def __init__(self, dir, data):
        # data numpy二维数组，
        # [('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O'), ('fname', 'O')] 
        # 前两项为：
        # [['39' '116' '569' '375' '14' '00001.jpg']
        # ['36' '116' '868' '587' '3' '00002.jpg']]
        self.dir = dir
        self.data = data
        print('数据集大小：', np.shape(data)[0])
    
    def __getitem__(self, index):
        Path = os.path.join(self.dir, self.data[index][-1]) # dir + '00001.jpg'
        bgr = cv2.imread(Path)
        x = [self.data[index][0], self.data[index][2]]
        y = [self.data[index][1], self.data[index][3]]
        label = self.data[index][4]
        
        # 对img_full各种处理，得img
        rgb = cv2.cvtColor(bgr,cv2.COLOR_RGB2BGR)
        # print(len(rgb.shape), rgb.shape, type(label))
        img = cv2.resize(rgb, (256, 256))
        # print(len(img.shape), img.shape, type(label))
        
        assert(len(img.shape)==3) # 注意有灰度图
        img = img.transpose([2,0,1]) # 3 * w * h
        return img, label
    
    def __len__(self):
        return np.shape(self.data)[0]


# 测试Dataset类：
import sys 
# print(sys.path)
sys.path.append(".")
# print(sys.path)
from UtilLib.Read_annos_mat import read_annos_to_np
if __name__ == '__main__':
    cars_train_annos_Path = './dataset/cars_train_annos.mat'
    img_dir = './dataset/cars_train'
    data_train = read_annos_to_np(cars_train_annos_Path)
    dataset = Dataset1(img_dir, data_train)
    trainLoader= torchData.DataLoader(dataset,batch_size=2,shuffle=False,drop_last=True)
    for data in trainLoader:
        imgs, labels = data
        for img in imgs:
            img=img.cpu().numpy() # tensor -> numpy
            img = img.transpose([1,2,0]) # 3 * w * h -> w * h * 3
            plt.figure()
            plt.imshow(img)
            plt.show()
        break
    print("测试结束！")