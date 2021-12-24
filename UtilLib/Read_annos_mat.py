import scipy.io as scio
import numpy as np

def read_annos_to_np(Path):
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