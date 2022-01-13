import os
import glob
import time
from testSubmit import run


class Args:

    def __init__(self) -> None:
        self.checkPointDir = 'log/Train_211227_2350/checkpoint'
        self.resultDir = 'log/Train_211227_2350/result'
        self.gpu_id = 5
        self.submit = True


args = Args()

checkpoints = glob.glob(args.checkPointDir + "/*.pth").sort()

# 并行测试并提交，可能爆显存
# max_num = 7
# num = 0
# for checkpoint in checkpoints:
#     num += 1
#     if num > max_num:
#         break
#     os.popen('CUDA_VISIBLE_DEVICES={} python testSubmit.py -m {} -p {} >> log/ranking.log'.format(args['gpu_id'], checkpoint, args['resultDir']))
#     time.sleep(10)

# 串行测试并提交
for checkpoint in checkpoints:
    args.model = checkpoint
    args.path = args.resultDir
    run(args)