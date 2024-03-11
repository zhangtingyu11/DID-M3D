import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius, center_to_corner_box3d
from lib.datasets.utils import draw_umich_gaussian, gen_gaussian_target
from lib.datasets.utils import get_angle_from_box3d,check_range
from lib.datasets.utils import points_cam2img
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import compute_box_3d
import pdb
import cv2 as cv
import torchvision.ops.roi_align as roi_align
import math
from lib.datasets.kitti_utils import Object3d

from shapely.geometry import MultiPoint, box
from typing import *
from numpy import random
import mmcv
import cv2



def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points
class KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.num_kpt  = 9
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        ''' 
        ##l,w,h
        self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                       [1.52563191462 ,1.62856739989, 3.88311640418],
                                       [1.73698127    ,0.59706367   , 1.76282397   ]])          

        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(root_dir, cfg['data_dir'], 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = os.path.join(root_dir, cfg['data_dir'], 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        self.dense_depth_dir = cfg['dense_depth_dir']

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4
        self.brightness_delta = 32
        self.contrast_lower, self.contrast_upper = (0.5, 1.5)
        self.saturation_lower, self.saturation_upper = (0.5, 1.5)
        self.hue_delta = 18

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode


    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        img = self.get_image(index)
        img_size = np.array(img.size)

        if self.split!='test':
            d = cv.imread('{}/{:0>6}.png'.format(self.dense_depth_dir, index), -1) / 256.
            dst_W, dst_H = img_size
            pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
            pad_wr = dst_W - pad_w - d.shape[1]
            d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
            d = Image.fromarray(d)


        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False

        if self.data_augmentation:
            # #! 增加新的数据增强方法
            # img = np.array(img)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # if random.randint(2):
            #     delta = random.uniform(-self.brightness_delta,
            #                         self.brightness_delta)
            #     img += delta

            # # mode == 0 --> do random contrast first
            # # mode == 1 --> do random contrast last
            # mode = random.randint(2)
            # if mode == 1:
            #     if random.randint(2):
            #         alpha = random.uniform(self.contrast_lower,
            #                             self.contrast_upper)
            #         img *= alpha

            # # convert color from BGR to HSV
            # img = mmcv.bgr2hsv(img)

            # # random saturation
            # if random.randint(2):
            #     img[..., 1] *= random.uniform(self.saturation_lower,
            #                                 self.saturation_upper)

            # # random hue
            # if random.randint(2):
            #     img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            #     img[..., 0][img[..., 0] > 360] -= 360
            #     img[..., 0][img[..., 0] < 0] += 360

            # # convert color from HSV to BGR
            # img = mmcv.hsv2bgr(img)

            # # random contrast
            # if mode == 0:
            #     if random.randint(2):
            #         alpha = random.uniform(self.contrast_lower,
            #                             self.contrast_upper)
            #         img *= alpha

            # # randomly swap channels
            # if random.randint(2):
            #     img = img[..., random.permutation(3)]
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                d = d.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        if self.split!='test':
            d_trans = d.transform(tuple(self.resolution.tolist()),
                                method=Image.AFFINE,
                                data=tuple(trans_inv.reshape(-1).tolist()),
                                resample=Image.BILINEAR)
            d_trans = np.array(d_trans)
            down_d_trans = cv.resize(d_trans, (self.resolution[0]//self.downsample, self.resolution[1]//self.downsample),
                            interpolation=cv.INTER_AREA)

        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)
        # image encoding
        # if random_crop_flag:
        #     cv2.imwrite("input_loader.png", np.array(img))
            # exit(0)
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W


        calib = self.get_calib(index)
        features_size = self.resolution // self.downsample# W * H
        #  ============================   get labels   ==============================
        if self.split!='test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi
            # labels encoding
            feat_w, feat_h = features_size
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            #! 增加关键点的预测
            center2kpt_offset_target = np.zeros([self.max_objs, self.num_kpt * 2], dtype=np.float32)
            kpt_heatmap_target = np.zeros([self.num_kpt, feat_h, feat_w], dtype=np.float32)
            kpt_heatmap_offset_target = np.zeros([self.max_objs, self.num_kpt * 2], dtype=np.float32)
            indices_kpt = np.zeros([self.max_objs, self.num_kpt], dtype=np.int64)
            mask_center2kpt_offset =  np.zeros([self.max_objs, self.num_kpt * 2], dtype=np.float32)
            mask_kpt_heatmap_offset = np.zeros([self.max_objs, self.num_kpt * 2], dtype=np.float32)
            
            # if torch.__version__ == '1.10.0+cu113':
            if torch.__version__ in ['1.10.0+cu113', '1.10.0', '1.6.0', '1.4.0']:
                mask_2d = np.zeros((self.max_objs), dtype=np.bool_)
            else:
                mask_2d = np.zeros((self.max_objs), dtype=np.bool_)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            vis_depth = np.zeros((self.max_objs, 7, 7), dtype=np.float32)
            att_depth = np.zeros((self.max_objs, 7, 7), dtype=np.float32)
            depth_mask = np.zeros((self.max_objs, 7, 7), dtype=np.bool_)

            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue
    
                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample


                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)

                if center_heatmap[0] < max(0, coord_range[0][0]/self.downsample) or center_heatmap[0] >= min(coord_range[1][0], features_size[0]): continue
                if center_heatmap[1] < max(0, coord_range[0][1]/self.downsample) or center_heatmap[1] >= min(coord_range[1][1], features_size[1]): continue
    
                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))
    
                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue
    
                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
    
                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h
    
                # encoding depth
                depth[i] = objects[i].pos[-1]
    
                # encoding heading angle
                #heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                #objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                    mask_2d[i] = 1

                roi_depth = roi_align(torch.from_numpy(down_d_trans).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                      [torch.tensor(bbox_2d).unsqueeze(0)], [7, 7]).numpy()[0, 0]
                # maintain interested points
                roi_depth_ind = (roi_depth > depth[i] - 3) & \
                                (roi_depth < depth[i] + 3) & \
                                (roi_depth > 0)
                roi_depth[~roi_depth_ind] = 0
                vis_depth[i] = roi_depth
                att_depth[i] = depth[i] - vis_depth[i]
                depth_mask[i] = roi_depth_ind

                #! 关键点数据
                loc = objects[i].pos[np.newaxis, :]
                dim = np.array([objects[i].l, objects[i].h, objects[i].w])[np.newaxis, :]
                rotation_y = np.array([objects[i].ry])[np.newaxis, :]
                dst = np.array([0.5, 0.5, 0.5])
                src = np.array([0.5, 1.0, 0.5])
                loc = loc + dim * (dst - src)
                offset = (calib.P2[0, 3] - calib.P0[0, 3]) \
                    / calib.P2[0, 0]
                # loc_3d = np.copy(loc)
                # loc_3d[0, 0] += offset
                gt_bbox_3d = np.concatenate([loc, dim, rotation_y], axis=1).astype(np.float32)
                corners_3d = center_to_corner_box3d(
                    gt_bbox_3d[:, :3],
                    gt_bbox_3d[:, 3:6],
                    gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
                    axis=1)
                corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
                all_corners_3d = corners_3d.copy()
                valid_corners_mask = np.zeros((8, 1))
        
                in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                corners_3d = corners_3d[:, in_front]

                # mark valid corners (depth > 0) as 1 (labelled but not visible, similar as COCO)
                valid_corners_mask[in_front, :] = 1

                # Project 3d box to 2d.
                camera_intrinsic = calib.P2
                camera_intrinsic = np.concatenate([camera_intrinsic, np.array([[0, 0, 0, 1]])], axis=0)
                corner_coords = view_points(corners_3d, camera_intrinsic,
                                            True).T[:, :2].tolist()
                corner_coords = [affine_transform(corner_coord, trans) for corner_coord in corner_coords]
                all_corner_coords = view_points(all_corners_3d, camera_intrinsic, True).T[:, :2]
                all_corner_coords = [affine_transform(corner_coord, trans) for corner_coord in all_corner_coords]
                all_corner_coords = np.hstack([all_corner_coords, valid_corners_mask])

                # Keep only corners that fall within the image.
                final_coords = post_process_coords(corner_coords, self.resolution)

                # Skip if the convex hull of the re-projected corners
                # does not intersect the image canvas.
                if final_coords is None:
                    continue
                
                center3d = np.array(objects[i].pos).reshape([1, 3])
                center2d = points_cam2img(
                    center3d, camera_intrinsic, with_depth=True)
                center2d[0][:2] = affine_transform(center2d[0][:2], trans)
                if center2d[0][2] <= 0:
                    continue
                center2d_valid = center2d.copy()
                center2d_valid[:, 2] = 1
                projected_pts = np.concatenate([all_corner_coords, center2d_valid])
                for kpt_idx in range(len(projected_pts)):
                    kptx, kpty, vis = projected_pts[kpt_idx]
                    is_kpt_in_image = (max(0, coord_range[0][0]) <= kptx <= min(coord_range[1][0], self.resolution[0])) and \
                        (max(0, coord_range[0][1]) <= kpty <= min(coord_range[1][1], self.resolution[1]))
                    if is_kpt_in_image:
                        projected_pts[kpt_idx, 2] = 2
                # if random_crop_flag:
                #     new_img = np.array(img)
                #     new_img = new_img.transpose(1, 2, 0) 
                #     new_img = (new_img*self.std) + self.mean
                #     new_img = (new_img*255).astype(np.uint8)
                #     flag = False
                #     for x, y, z in projected_pts:
                #         if (0 <= x <= self.resolution[0]) and (0 <= y <= self.resolution[1]) and ((x < coord_range[0][0]) or (x > coord_range[1][0])):
                #             flag = True
                #         new_img = cv2.circle(new_img, (int(x), int(y)), 3, (0, 255, 0), 2)
                #     if flag:
                #         cv2.imwrite("affine.png", new_img)
                #         exit(0)
                kpts_2d = np.array(projected_pts.reshape(-1, 3))
                kpts_valid_mask = kpts_2d[:, 2].astype('int64')
                kpts_2d = kpts_2d[:, :2].astype('float32').reshape(-1)
                kpts_2d = kpts_2d.reshape(self.num_kpt, 2)
                kpts_2d[:, 0] /= self.downsample
                kpts_2d[:, 1] /= self.downsample
                ctx_int, cty_int = center_2d.astype(np.int32)
                ctx, cty = center_2d
                scale_box_h = (bbox_2d[3] - bbox_2d[1])
                scale_box_w = (bbox_2d[2] - bbox_2d[0])
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                for k in range(self.num_kpt):
                    kpt = kpts_2d[k]
                    kptx_int, kpty_int = int(kpt[0]), int(kpt[1])
                    kptx, kpty = kpt
                    vis_level = kpts_valid_mask[k]
                    if vis_level < 1:
                        continue

                    center2kpt_offset_target[i, k * 2] = kptx - ctx_int
                    center2kpt_offset_target[i, k * 2 + 1] = kpty - cty_int
                    mask_center2kpt_offset[i, k * 2:k * 2 + 2] = 1

                    is_kpt_inside_image = (max(0, coord_range[0][0]/self.downsample) <= kptx_int < min(coord_range[1][0]/self.downsample, feat_w)) and (max(0, coord_range[0][1]/self.downsample) <= kpty_int < min(coord_range[1][1]/self.downsample, feat_h))
                    if not is_kpt_inside_image:
                        continue

                    draw_umich_gaussian(kpt_heatmap_target[k],
                                        [kptx_int, kpty_int], radius)

                    kpt_index = kpty_int * feat_w + kptx_int
                    indices_kpt[i, k] = kpt_index

                    kpt_heatmap_offset_target[i, k * 2] = kptx - kptx_int
                    kpt_heatmap_offset_target[i, k * 2 + 1] = kpty - kpty_int
                    mask_kpt_heatmap_offset[i, k * 2:k * 2 + 2] = 1

            targets = {'depth': depth,
                       'size_2d': size_2d,
                       'heatmap': heatmap,
                       'offset_2d': offset_2d,
                       'indices': indices,
                       'size_3d': size_3d,
                       'offset_3d': offset_3d,
                       'heading_bin': heading_bin,
                       'heading_res': heading_res,
                       'cls_ids': cls_ids,
                       'mask_2d': mask_2d,

                       'vis_depth': vis_depth,
                       'att_depth': att_depth,
                       'depth_mask': depth_mask,
                       "center2kpt_offset_target" : center2kpt_offset_target,
                       "kpt_heatmap_target": kpt_heatmap_target,
                       "kpt_heatmap_offset_target": kpt_heatmap_offset_target,
                       "mask_center2kpt_offset": mask_center2kpt_offset,
                       "indices_kpt": indices_kpt,
                       "mask_kpt_heatmap_offset": mask_kpt_heatmap_offset
                       }
        else:
            targets = {}

        inputs = img
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}

        return inputs, calib.P2, coord_range, targets, info   #calib.P2


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'random_flip':0.0, 'random_crop':1.0, 'scale':0.4, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False}
    dataset = KITTI('../../data', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.bool_))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
