import os
from PIL import Image
import cv2
import numpy as np

import sys
sys.path.append('..')
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

class DrawCorners():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, "image_2")
        self.label_dir = os.path.join(self.root_dir, "label_2")
        self.calib_dir = os.path.join(self.root_dir, "calib")
        
    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)
    
    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)
    
    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)
    
    def compute_box_3d(self, obj, P):
        """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
        """
        # compute rotational matrix around yaw axis
        R = roty(obj.ry)

        # 3d bounding box dimensions
        l = obj.l
        w = obj.w
        h = obj.h

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + obj.pos[0]
        corners_3d[1, :] = corners_3d[1, :] + obj.pos[1]
        corners_3d[2, :] = corners_3d[2, :] + obj.pos[2]
        # print 'cornsers_3d: ', corners_3d
        # only draw 3d bounding box for objs in front of the camera
        if np.any(corners_3d[2, :] < 0.1):
            corners_2d = None
            return corners_2d, np.transpose(corners_3d)

        # project the 3d bounding box into the image plane
        corners_2d = self.project_to_image(np.transpose(corners_3d), P)
        # print 'corners_2d: ', corners_2d
        return corners_2d, np.transpose(corners_3d)
    
    def project_to_image(self, pts_3d, P):
        """ Project 3d points to image plane.

        Usage: pts_2d = projectToImage(pts_3d, P)
        input: pts_3d: nx3 matrix
                P:      3x4 projection matrix
        output: pts_2d: nx2 matrix

        P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
        => normalize projected_pts_2d(2xn)

        <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
            => normalize projected_pts_2d(nx2)
        """
        n = pts_3d.shape[0]
        pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
        # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
        pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]
    
    def draw_projected_box3d(self, image, qs, color=(0, 255, 0), thickness=2):
        """ Draw 3d bounding box in image
            qs: (8,3) array of vertices for the 3d box in following order:
                1 -------- 0
            /|         /|
            2 -------- 3 .
            | |        | |
            . 5 -------- 4
            |/         |/
            6 -------- 7
        """
        qs = qs.astype(np.int32)
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        return image
    
if __name__ == "__main__":
    dc = DrawCorners("/home/zty/Project/DeepLearning/DID-M3D/data/KITTI3D/training")
    img = dc.get_image(8)
    calib = dc.get_calib(8)
    objs = dc.get_label(8)
    thickness = 2
    for obj in objs:
        box3d_pts_2d, _ = dc.compute_box_3d(obj, calib.P2)
        if box3d_pts_2d is None:
            continue
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = dc.draw_projected_box3d(img, box3d_pts_2d, color=(0, 255, 0), thickness=thickness)
    cv2.imwrite("corners.png", img)
        # break