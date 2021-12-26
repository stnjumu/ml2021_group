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