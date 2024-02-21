import pickle
import numpy as np
import matplotlib.pyplot as plt

def read_pickle_file(filename):
    with open(filename, 'rb') as f:
        pickle_info = pickle.load(f)
    return pickle_info
def func(x):
    return -2 * np.log(np.sqrt(2)/(2*x))

def draw_last_epoch_relation(pickle_info):
    last_epoch_info = pickle_info[-1]
    total_vis_depth_offset = np.empty(0)
    total_att_depth_offset = np.empty(0)
    total_vis_depth_uncer_pred = np.empty(0)
    total_att_depth_uncer_pred = np.empty(0)
    total_depth_target = np.empty(0)
    total_ins_depth_offset = np.empty(0)
    total_new_ins_depth_offset = np.empty(0)
    
    for idx, batch_info in enumerate(last_epoch_info):
        vis_depth_pred = batch_info["vis_depth_pred"]
        att_depth_pred = batch_info["att_depth_pred"]
        ins_depth_pred = vis_depth_pred + att_depth_pred
        vis_depth_target = batch_info["vis_depth_target"]
        att_depth_target = batch_info["att_depth_target"]
        vis_depth_uncer_pred = batch_info["vis_depth_uncertain_pred"]
        att_depth_uncer_pred = batch_info["att_depth_uncertain_pred"]
        ins_depth_uncer_pred = batch_info["ins_depth_uncertain_pred"]
        
        merge_prob = np.exp((-np.exp((0.5 * ins_depth_uncer_pred))))
        new_merge_prob = np.zeros_like(merge_prob)
        merge_mask = merge_prob > 0.5
        new_merge_prob[merge_mask] = merge_prob[merge_mask]
        invalid_merge_mask = np.sum(merge_mask.reshape(merge_mask.shape[0], 49), axis = -1)==0
        new_merge_prob[invalid_merge_mask] = merge_prob[invalid_merge_mask]
        new_merge_depth = (np.sum((ins_depth_pred*new_merge_prob).reshape(-1, 49), axis=-1) /
                (np.sum(new_merge_prob.reshape(-1, 49), axis=-1)+1e-8))
        new_merge_depth = new_merge_depth.reshape(-1)
        merge_depth = (np.sum((ins_depth_pred*merge_prob).reshape(-1, 49), axis=-1) /
                (np.sum(merge_prob.reshape(-1, 49), axis=-1)+1e-8))
        merge_depth = merge_depth.reshape(-1)
        
        depth_target = batch_info["depth_target"].reshape(-1)
        total_ins_depth_offset = np.concatenate([total_ins_depth_offset, merge_depth-depth_target])
        total_new_ins_depth_offset = np.concatenate([total_new_ins_depth_offset, new_merge_depth-depth_target])
        total_depth_target = np.concatenate([total_depth_target, depth_target])
        # vis_depth_offset = np.abs(vis_depth_pred - vis_depth_target)
        # att_depth_offset = np.abs(att_depth_pred - att_depth_target)
        # total_vis_depth_offset = np.concatenate([total_vis_depth_offset, vis_depth_offset])
        # total_att_depth_offset = np.concatenate([total_att_depth_offset, att_depth_offset])
        # total_vis_depth_uncer_pred = np.concatenate([total_vis_depth_uncer_pred, vis_depth_uncer_pred])
        # total_att_depth_uncer_pred = np.concatenate([total_att_depth_uncer_pred, att_depth_uncer_pred])
    # plt.scatter(total_vis_depth_offset, total_vis_depth_uncer_pred, s=0.1)
    plt.scatter(total_depth_target, total_ins_depth_offset, s=0.1)
    total_ins_depth_offset = np.abs(total_ins_depth_offset)
    total_new_ins_depth_offset = np.abs(total_new_ins_depth_offset)
    better = np.sum(total_new_ins_depth_offset < total_ins_depth_offset-0.1)
    worse = np.sum(total_new_ins_depth_offset > total_ins_depth_offset+0.1)
    print(better, worse)
    xs = np.arange(0.1, 70, 0.1)
    ys = np.array([func(item) for item in xs])
    plt.plot(xs,ys, color='r')
    # plt.savefig("function.png")
    plt.savefig("lab/ins_depth_offset_epoch150.png")
    
if __name__ == "__main__":
    pickle_info = read_pickle_file("/home/public/zty/Project/DeepLearningProject/DID-M3D/record_150.pkl")
    draw_last_epoch_relation(pickle_info)
        
        
        

        
        