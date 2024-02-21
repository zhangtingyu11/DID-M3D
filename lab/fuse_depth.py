import cv2
from PIL import Image
import numpy as np

def read_depth_img_and_raw_image(index):
    raw_image = cv2.imread('{}/{:0>6}.png'.format("/home/public/zty/Project/DeepLearningProject/DID-M3D/data/KITTI3D/training/image_2", index), -1) / 256.
    img_size = raw_image.shape[:2]
    d = cv2.imread('{}/{:0>6}.png'.format("/home/public/zty/Project/DeepLearningProject/DID-M3D/data/KITTI3D/training/depth_dense_lrru_my_version_2.0clip", index), -1) / 256.
    dst_H, dst_W = img_size
    pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
    pad_wr = dst_W - pad_w - d.shape[1]
    d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
    d = Image.fromarray(d)
    
if __name__ == "__main__":
    read_depth_img_and_raw_image(0)