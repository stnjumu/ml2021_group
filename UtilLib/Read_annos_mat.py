import scipy.io as scio
import numpy as np

def read_annos_to_np(Path):
    """读取cars_train_annos.mat或cars_test_annos.mat文件，转换为二维numpy数组

    Args:
        Path (str): 文件路径

    Returns:
        numpy二维数组:  训练集: 8144*6
                        测试集: 8041*5
    """
    cars_annos = scio.loadmat(Path)
    annotations = cars_annos['annotations'][0]
    for annotation in annotations:
        for item in annotation:
            # print(np.squeeze(item)) # 一个数或一个string，但type = <class 'numpy.ndarray'>
            shape = np.shape(np.squeeze(item)) # 应为()
            assert(len(shape)==0)
            # 注：item[0][0]是前5项int值，item[0]是第6项str
    # break
    data = np.array([np.array([np.squeeze(item) for item in annotation]) for annotation in annotations])
    return data