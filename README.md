# ml2021_group

数据集放在./dataset下，.gitignore已忽略该文件夹

# 关键修改写在下面：
## 2021年12月24日
    mu: 创建项目，数据集类放在DatasetLib文件夹下，模型类放在ModelLib中，通用函数定义放在UtilLib中；
    trainValid.py在测试集上进行训练+验证；
    代码还跑不起来，可能有各种bug；
## 2021年12月25日
    mu: 修复bug，代码跑起来了
    创建两个分支尝试新模型       efficientNet: ./DatasetLib/automl-master/efficientnetv2/effnetv2_model.py
                    Tresnet: ./DatasetLib/ImageNet21K-main/src_files/models/tresnet/tresnet.py
## 2021年12月26, 27日
    最好test acc：74.5%
## 2021年12月27日
    最好test acc: 94%
    liangwei:TResnet、增加自动数据增强、增加批量提交代码，trainValid整理


# TODO:
## 更新框架代码
### 1. 保存checkpoint添加优化器
### 2. 训练, 验证，预测，提交
### 3. 改进保存checkpoint逻辑，用提交成绩判断
### 4. 改进参数字典，优化器参数，切分用random_split，Dataloader放外边