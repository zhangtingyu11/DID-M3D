import pickle
from lib.losses.focal_loss import focal_loss_cornernet_with_boundary as focal_loss
import torch
import cv2
import numpy as np
with open('train_phase.pkl', 'rb') as f:
    info = pickle.load(f)
    boundary = torch.from_numpy(info["target"]["boundary"]).cuda()
    pred_kpt_heatmap = torch.from_numpy(info['output']['kpt_heatmap']).cuda()
    target_kpt_heatmap = torch.from_numpy(info['target']['kpt_heatmap_target']).cuda()
    loss = focal_loss(pred_kpt_heatmap, target_kpt_heatmap, boundary)
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1, 1, -1)
    std  = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1, 1, -1)
    for bs in range(16):
        pkh = pred_kpt_heatmap[bs]
        tkh = target_kpt_heatmap[bs]
        input = torch.from_numpy(info["input"][bs]).cuda().permute(1, 2, 0)
        input = ((input*std+mean) * 255).int()
        cv2.imwrite("kpt_heatmap/pic{}.png".format(bs), np.array(input.detach().cpu()))
        pass