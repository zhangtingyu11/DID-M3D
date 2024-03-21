import cv2
from PIL import Image
import numpy as np
import sys
import torch
sys.path.append('../..')
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
        dst_W, dst_H = img_size
        center = np.array(img_size) / 2
        crop_size = img_size
        
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        raw_image = raw_image.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        
        
        raw_image = np.array(raw_image)
        bboxes2d = self.get_bbox2d(index)
        for i, bbox2d in enumerate(bboxes2d[1:]):
            bbox_2d = bbox2d.copy()
            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
            # modify the 2d bbox according to pre-compute downsample ratio
            bbox_2d_int = bbox_2d.astype(np.int32)
            bgr_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
            object_instance = bgr_image[bbox_2d_int[1]:bbox_2d_int[3], bbox_2d_int[0]:bbox_2d_int[2], :].copy()
            cv2.imwrite("cropped_object_instance.png", object_instance)
            
            rec_img = cv2.rectangle(bgr_image.copy(), (bbox_2d_int[0], bbox_2d_int[1]), (bbox_2d_int[2], bbox_2d_int[3]), color = (255, 0, 0),
                                    thickness = 2)
            rec_img = cv2.rectangle(rec_img, (bbox_2d_int[0], bbox_2d_int[1]), (bbox_2d_int[2], bbox_2d_int[3]), color = (0, 0, 255),
                        thickness = 2)
            cv2.imwrite("rec_img.png", rec_img)
            w = bbox_2d_int[2] - bbox_2d_int[0]
            h = bbox_2d_int[3] - bbox_2d_int[1]
            center_x = (bbox_2d_int[2] + bbox_2d_int[0])/2
            center_y = (bbox_2d_int[3] + bbox_2d_int[1])/2
            
            for added_pixel in [10, 20]:
                scale_w = w+added_pixel
                scale_h = h+added_pixel
                bbox_2d_int_new = np.array([center_x-scale_w/2, center_y-scale_h/2, center_x+scale_w/2, center_y+scale_h/2], dtype = np.int32)
                object_instance = bgr_image[bbox_2d_int_new[1]:bbox_2d_int_new[3], bbox_2d_int_new[0]:bbox_2d_int_new[2], :].copy()
                cv2.imwrite("cropped_object_instance_{}.png".format(added_pixel), object_instance)
            
                rec_img = cv2.rectangle(bgr_image.copy(), (bbox_2d_int_new[0], bbox_2d_int_new[1]), (bbox_2d_int_new[2], bbox_2d_int_new[3]), color = (0, 0, 255),
                                        thickness = 2)
                rec_img = cv2.rectangle(rec_img, (bbox_2d_int[0], bbox_2d_int[1]), (bbox_2d_int[2], bbox_2d_int[3]), color = (255, 0, 0),
                            thickness = 2)
                cv2.imwrite("rec_img_{}.png".format(added_pixel), rec_img)
                
            
        
            
            

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
    drdg.read_depth_img_and_raw_image(35)