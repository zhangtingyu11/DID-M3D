import cv2
from PIL import Image
import numpy as np
import torchvision.ops.roi_align as roi_align

def read_depth_img_and_raw_image(index):
    raw_image = cv2.imread('{}/{:0>6}.png'.format("/home/public/zty/Project/DeepLearningProject/DID-M3D/data/KITTI3D/training/image_2", index), -1)
    img_size = raw_image.shape[:2]
    d = cv2.imread('{}/{:0>6}.png'.format("/home/public/zty/Project/DeepLearningProject/DID-M3D/data/KITTI3D/training/depth_dense_lrru_my_version_2.0clip", index), -1) / 256.
    dst_H, dst_W = img_size
    pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
    pad_wr = dst_W - pad_w - d.shape[1]
    d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
    d = d.astype(np.uint8)
    d_max = d.max()
    d_min = d.min()
    d = ((d-d_min)/(d_max-d_min) * 255).astype(np.uint8)
    depth_map_colored = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    down_depth_map_colored = cv2.resize(depth_map_colored, (dst_W//4, dst_H//4),
                            interpolation=cv2.INTER_AREA)
    down_raw_image = cv2.resize(raw_image, (dst_W//4, dst_H//4),
                        interpolation=cv2.INTER_AREA)
    alpha = 0.8
    overlay = cv2.addWeighted(down_raw_image, 1 - alpha, down_depth_map_colored, alpha, 0)
    
    # 显示结果
    # cv2.imshow("Overlay", overlay)
    cv2.imwrite("overlay.png", down_depth_map_colored)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    read_depth_img_and_raw_image(3)