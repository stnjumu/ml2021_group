# 数据集示例
# 只是读取图片，裁剪后resize到256*256
import torch
import torch.utils.data as torchData
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from torchvision.transforms.transforms import Normalize
class DatasetTorch(torchData.Dataset):
    def __init__(self, dir, data, split='train', aug=True, img_size=[350, 350], img_mode = "RGB"):
        # data numpy二维数组，
        # [('bbox_x1', 'O'), ('bbox_y1', 'O'), ('bbox_x2', 'O'), ('bbox_y2', 'O'), ('class', 'O'), ('fname', 'O')] 
        # 前两项为：
        # [['39' '116' '569' '375' '14' '00001.jpg']
        # ['36' '116' '868' '587' '3' '00002.jpg']]
        self.dir = dir
        self.img_mode = img_mode

        # 训练数据集增强
        self.split = split
        self.aug = aug
        if self.aug and self.split=='train':
            self.tfs = transforms.Compose([
                 #transforms.RandomSizedCrop((img_size[0], img_size[1])),
                 transforms.Resize(img_size),
                 transforms.RandomGrayscale(p = 0.05),
                 transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.data = data
            self.tfs = transforms.Compose([
                 transforms.Resize(img_size),
                 transforms.ToTensor()
            ])
    
    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.data[index][-1]) # dir + '00001.jpg'
        x = [int(self.data[index][0]), int(self.data[index][2])]
        y = [int(self.data[index][1]), int(self.data[index][3])]
        img = Image.open(img_path).convert(self.img_mode)
        
        img = self.tfs(img.crop([x[0],y[0],x[1],y[1]]))
        if self.img_mode == "L": # 灰度图在第0维，即channel维恢复为3维度
            img = torch.cat((img, img, img), dim=0)

        if self.split == 'test':
            path = self.data[index][-1]
            return img, path
        label = self.data[index][4]
        label = torch.from_numpy(np.array(int(label))).long() - 1
        img = img.cuda()
        label = label.cuda()

        return img, label
    

    
    def __len__(self):
        return np.shape(self.data)[0]


# 测试Dataset类：
import sys 
# print(sys.path)
sys.path.append("..")
#print(sys.path)
from UtilLib.Read_annos_mat import read_annos_to_np
if __name__ == '__main__':
    cars_train_annos_Path = '../dataset/cars_train_annos.mat'
    img_dir = '../dataset/cars_train'
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