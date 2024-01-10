from collections import defaultdict
import numpy as np
origin_log_file = "/home/public/zty/Project/DeepLearningProject/DID-M3D/kitti_models/logs/no_filter_trun_and_occ/train.log"
other_log_file = "/home/public/zty/Project/DeepLearningProject/DID-M3D/kitti_models/logs/transformer_encoder1/train.log"

def get_loss_dict(filename):
    loss_dict = defaultdict(list)
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "BATCH[0200/0232]" in line:
                index = line.find("BATCH[0200/0232]")
                line = line[index+16:]
                splits = line.split(',')
                for sp in splits[:-1]:
                    loss_name, loss_val = sp.split(':')
                    loss_name = loss_name.strip()
                    loss_val = float(loss_val)
                    loss_dict[loss_name].append(loss_val)
    return loss_dict

if __name__ == "__main__":
    origin_loss_dict = get_loss_dict(origin_log_file)
    other_loss_dict = get_loss_dict(other_log_file)
    
    origin_len = len(origin_loss_dict["depth_loss"])
    other_len = len(other_loss_dict["depth_loss"])
    
    mn_len = min(origin_len, other_len)
    for key, val in origin_loss_dict.items():
        origin_loss_dict[key] = val[:mn_len]
    for key, val in other_loss_dict.items():
        other_loss_dict[key] = val[:mn_len]
        
    interval_loss_dicts = []
    interval = 30
    for start_idx in range(0, 150, interval):
        interval_loss_dict = defaultdict(float)
        for key in origin_loss_dict.keys():
            origin_loss = np.array(origin_loss_dict[key][start_idx:start_idx+interval])
            other_loss = np.array(other_loss_dict[key][start_idx:start_idx+interval])
            offset = origin_loss-other_loss
            interval_loss_dict[key] = np.mean(offset)
        interval_loss_dicts.append(interval_loss_dict)
    for item in interval_loss_dicts:
        print(item)
            
            
            
        
                    