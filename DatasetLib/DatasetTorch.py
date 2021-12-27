# 实现1
import torch
import torch.utils.data as torchData
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from .auto_augment import AutoAugment, ImageNetAutoAugment
class DatasetTorch(torchData.Dataset):
    def __init__(self, dir, data, split='train', aug=True, img_size=[384,384]):
        # data numpy二维数组，
        # [('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O'), ('fname', 'O')] 
        # 前两项为：
        # [['39' '116' '569' '375' '14' '00001.jpg']
        # ['36' '116' '868' '587' '3' '00002.jpg']]
        self.dir = dir
        
        
        # 训练数据集增强
        self.split = split
        self.aug = aug
        if self.aug and self.split=='train':
            # self.data =  np.concatenate((data, data, data), axis=0)
            self.data = data
            self.tfs = transforms.Compose([
                 transforms.Resize((img_size[0], img_size[1])),
                 ImageNetAutoAugment(),
                 transforms.ToTensor(),
                #  transforms.Normalize(
                #     mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5]),
            ])
        else:
            self.data = data
            self.tfs = transforms.Compose([
                 transforms.Resize((img_size[0], img_size[1])),
                 transforms.ToTensor(),
                #  transforms.Normalize(
                #     mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5]),
            ])
            
        print('数据集大小：', np.shape(data)[0])

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.data[index][-1]) # dir + '00001.jpg'
        x = [int(self.data[index][0]), int(self.data[index][2])]
        y = [int(self.data[index][1]), int(self.data[index][3])]
        img = Image.open(img_path).convert("RGB")
        img = self.tfs(img.crop([x[0],y[0],x[1],y[1]]))

        if self.split != 'test':
            label = self.data[index][4]
            label = torch.from_numpy(np.array(int(label))).long()-1
            return img, label
        else:
            return img, str(self.data[index][-1])
    
    
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
    dataset = DatasetTorch(img_dir, data_train)
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