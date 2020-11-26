'''
ml load anaconda
conda activate ke_seg
cd /home/nfs/em4/shuchangyong/submit

python val.py --data-dir /home/nfs/em2/wei_dataset/raw_datasets/CITYSCAPES/data/cityscapes  --restore-from new_rn18-cityscape_singleAndWhole_val-75.90_test-74.58.pth --gpu 0 --type val  --figsavepath 75.9val
python val.py --data-dir /home/nfs/em2/wei_dataset/raw_datasets/CITYSCAPES/data/cityscapes  --restore-from new_rn18-cityscape_singleAndWhole_val-75.90_test-74.58.pth --gpu 2 --type test --figsavepath 74.58test

python val.py --data-dir /home/nfs/em2/wei_dataset/raw_datasets/CITYSCAPES/data/cityscapes  --restore-from new_rn18-cityscape_singleAndWhole_val-75.15_test-74.18.pth --gpu 1 --type val  --figsavepath 75.15val
python val.py --data-dir /home/nfs/em2/wei_dataset/raw_datasets/CITYSCAPES/data/cityscapes  --restore-from new_rn18-cityscape_singleAndWhole_val-75.15_test-74.18.pth --gpu 3 --type test --figsavepath 74.18test

python val.py --data-dir /home/nfs/em2/wei_dataset/raw_datasets/CITYSCAPES/data/cityscapes  --restore-from new_rn18-cityscape_singleAndWhole_val-75.02_test-00.00.pth --gpu 1 --type val  --figsavepath 75.02val
python val.py --data-dir /home/nfs/em2/wei_dataset/raw_datasets/CITYSCAPES/data/cityscapes  --restore-from new_rn18-cityscape_singleAndWhole_val-75.02_test-00.00.pth --gpu 2 --type test  --figsavepath 00.00test

'''

import os
import torch
from options import ValOptions
from torch.utils import data
from dataset.datasets import CSTrainValSet, CSTestSet
from networks.pspnet import Res_pspnet, BasicBlock, Bottleneck
from utils.evaluate import evaluate_main
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = ValOptions().initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.type == 'val':
        loader = data.DataLoader(CSTrainValSet(args.data_dir, args.val_data_list, crop_size=(1024, 2048), scale=False, mirror=False), batch_size=1, shuffle=False, pin_memory=True)
    elif args.type == 'test':
        loader = data.DataLoader(CSTestSet(args.data_dir, args.test_data_list, crop_size=(1024, 2048)), batch_size=1, shuffle=False, pin_memory=True)

    student = Res_pspnet(BasicBlock, [2, 2, 2, 2], num_classes = args.num_classes)
    student.load_state_dict(torch.load(args.restore_from))
    print("=> load " + str(args.restore_from))

    mean_IU, IU_array = evaluate_main(args.figsavepath, student, loader, args.num_classes, args.type)
    print('mean_IU: {:.6f}  IU_array: \n{}'.format(mean_IU, IU_array))
