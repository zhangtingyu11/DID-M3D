from lib.datasets.kitti import KITTI
import yaml
import numpy as np

def check(ry, limit):
    return -limit < ry < limit or np.pi-limit < ry < np.pi + limit

if __name__ == "__main__":
    cfg = yaml.load(open('config/kitti.yaml', 'r'), Loader=yaml.Loader)
    kitti = KITTI('./', 'train', cfg['dataset'])
    offset_list = []
    for frame_idx in kitti.idx_list:
        frame_idx = int(frame_idx)
        label = kitti.get_label(frame_idx)
        calib = kitti.get_calib(frame_idx)
        fv = calib.fv
        for item in label:
            h_3d = item.h
            box2d = item.box2d
            h_2d = box2d[3]-box2d[1]
            depth_pred = fv * h_3d / h_2d
            depth = item.pos[2]
            ry = item.ry
            l, w = item.l, item.w
            limit = np.arctan2(l, w)
            offset = abs(w/2 / (np.cos(ry)+1e-8)) * check(ry, limit) + abs(l/2 / (np.sin(ry)+1e-8)) * (1-check(ry, limit))
            if item.cls_type not in kitti.writelist or item.level_str=='UnKnown':
                continue
            
            offset_list.append(depth-depth_pred-offset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          )
    offset = np.array(offset_list)
    print(np.mean(offset))
    print(np.var(offset))
    print(np.max(offset))
    print(np.min(offset))
    
    
            