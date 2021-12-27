import os
import glob
from testSubmit import run
args = {
    'checkPointDir' : 'log/Train_211227_1652/checkpoint',
    'resultDir' : 'log/Train_211227_1652/result',
    'gpu_id' : 5,
    'submit' : True 
}

checkpoints = glob.glob(args['checkPointDir']+"/*.pth")

# 并行测试并提交，可能爆显存
max_num = 7
num = 0 
for checkpoint in checkpoints: 
    num += 1
    if num > max_num:
        break
    os.popen('CUDA_VISIBLE_DEVICES={} python testSubmit.py -m {} -p {} >> log/ranking.log'.format(args['gpu_id'], checkpoint, args['resultDir']))

# 串行测试并提交
for checkpoint in checkpoints: 
    args['checkPointDir'] = checkpoint
    run(args)