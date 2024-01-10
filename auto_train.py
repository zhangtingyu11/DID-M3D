import os
os.system('CUDA_VISIBLE_DEVICES=0,1 python tools/train_val.py --config config/kitti.yaml')
os.system('CUDA_VISIBLE_DEVICES=0,1 python tools/train_val.py --config config/kitti_new.yaml')
