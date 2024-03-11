import cv2
from PIL import Image
import numpy as np
import sys
import torch
sys.path.append('..')
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import get_affine_transform
import os
import cv2 as cv
import torchvision.ops.roi_align as roi_align
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
class DrawRoiDepthGT:
    def __init__(self):
        self.resolution = np.array([1280, 384])
        self.downsample = 4
        root_dir = '/home/zty/Project/DeepLearning/DID-M3D'
        self.image_dir = os.path.join(root_dir, "data/KITTI3D/training/image_2")
        self.label_dir = os.path.join(root_dir, "data/KITTI3D/training/label_2")
        
        
    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def read_depth_img_and_raw_image(self, index):
        raw_image = self.get_image(index)
        img_size = np.array(raw_image.size)
        d = cv2.imread('{}/{:0>6}.png'.format("/home/zty/Project/DeepLearning/DID-M3D/data/KITTI3D/training/depth_dense_lrru_my_version_2.0clip", index), -1) / 256.
        dst_W, dst_H = img_size
        center = np.array(img_size) / 2
        crop_size = img_size
        
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        raw_image = raw_image.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        
        pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
        pad_wr = dst_W - pad_w - d.shape[1]
        d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
        d = Image.fromarray(d)
        d_trans = d.transform(tuple(self.resolution.tolist()),
                    method=Image.AFFINE,
                    data=tuple(trans_inv.reshape(-1).tolist()),
                    resample=Image.BILINEAR)
        d_trans = np.array(d_trans)
        down_d_trans = cv.resize(d_trans, (self.resolution[0]//self.downsample, self.resolution[1]//self.downsample),
                        interpolation=cv.INTER_AREA)
        raw_image = np.array(raw_image)
        down_raw_image = cv.resize(raw_image, (self.resolution[0]//self.downsample, self.resolution[1]//self.downsample),
                interpolation=cv.INTER_AREA)
        
        bboxes2d = self.get_bbox2d(index)
        for i, bbox2d in enumerate(bboxes2d):
            bbox_2d = bbox2d.copy()
            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            # modify the 2d bbox according to pre-compute downsample ratio
            bbox_2d[:] /= self.downsample
            bbox_2d_int = bbox_2d.astype(np.int32)
            bgr_image = cv2.cvtColor(down_raw_image, cv2.COLOR_RGB2BGR)
            object_instance = bgr_image[bbox_2d_int[1]:bbox_2d_int[3], bbox_2d_int[0]:bbox_2d_int[2], :]
            cropped_depth = down_d_trans[bbox_2d_int[1]:bbox_2d_int[3], bbox_2d_int[0]:bbox_2d_int[2]]
            roi_depth = roi_align(torch.from_numpy(down_d_trans).unsqueeze(0).unsqueeze(0).type(torch.float32),
                        [torch.tensor(bbox_2d).unsqueeze(0)], [7, 7]).numpy()[0, 0]
            cv2.imwrite("cropped_object_instance.png", object_instance)
            cv2.imwrite("cropped_depth.png", cropped_depth)
            center_depth = roi_depth[3, 3]
            depth_offset = np.abs(center_depth-roi_depth)
            
            cmap = plt.cm.RdBu  # 选择颜色映射，这里使用红蓝颜色映射
            norm = Normalize(vmin=np.min(depth_offset), vmax=np.max(depth_offset))

            # 创建图像并显示
            plt.imshow(depth_offset, cmap=cmap, norm=norm, interpolation='nearest')
            cb = plt.colorbar()

            tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
            cb.locator = tick_locator
            cb.set_ticks([np.min(depth_offset), np.max(depth_offset)])
            cb.update_ticks()
            cb.ax.tick_params(size=0)
            # 可选：在每个像素位置显示深度值
            for i in range(depth_offset.shape[0]):
                for j in range(depth_offset.shape[1]):
                    plt.text(j, i, f'{depth_offset[i, j]:.2f}', color='black',
                            ha='center', va='center', fontsize=8)
            plt.axis('off') # 去坐标轴
            plt.xticks([]) # 去刻度
            plt.yticks([]) # 去刻度
            plt.savefig("visual_depth_2.png", bbox_inches='tight', pad_inches = 0)
            break
        
    def get_bbox2d(self, index):
        labels = self.get_label(index)
        bboxes2d = []
        for label in labels:
            bbox2d = label.box2d
            bboxes2d.append(bbox2d)
        return bboxes2d
        
    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)
    
if __name__ == "__main__":
    drdg = DrawRoiDepthGT()
    drdg.read_depth_img_and_raw_image(7461)