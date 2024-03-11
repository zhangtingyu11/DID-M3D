import pickle
import numpy as np
import matplotlib.pyplot as plt

def read_pickle_file(filename):
    """读取pickle文件

    Args:
        filename (_type_): 文件路径

    Returns:
        _type_: 读取到的数据
    """
    with open(filename, 'rb') as f:
        pickle_info = pickle.load(f)
    return pickle_info


def min_uncertainty_loss_log_variance(x):
    """对于uncertainty函数, 将abs(pred-gt)看成常量, log_variance看成唯一变量时, loss取最小值时log_variance的取值

    Args:
        x (_type_): abs(pred-gt)的取值

    Returns:
        _type_: loss取最小值时log_variance的取值
    """
    return -2 * np.log(np.sqrt(2)/(2*x))

def draw_func(func, xs):
    """画出函数图像

    Args:
        func (_type_): 函数
        xs (_type_): 横坐标的数值
    """
    ys = np.array([func(item) for item in xs])
    plt.plot(xs,ys, color='r')
    
def get_merge_prob_origin(ins_depth_uncer_pred):
    """论文中原始根据不确定性得到深度的可能性

    Args:
        ins_depth_uncer_pred (_type_): 深度的不确定性

    Returns:
        _type_: 深度的可能性
    """
    merge_prob = np.exp((-np.exp((0.5 * ins_depth_uncer_pred))))
    return merge_prob

def get_merge_prob_filter(ins_depth_uncer_pred):
    """根据不确定性得到深度的可能性(过滤太低的可能性的情况)

    Args:
        ins_depth_uncer_pred (_type_): 深度的不确定性

    Returns:
        _type_: 深度的可能性
    """
    merge_prob = np.exp((-np.exp((0.5 * ins_depth_uncer_pred))))
    new_merge_prob = np.zeros_like(merge_prob)
    merge_mask = merge_prob > 0.5
    new_merge_prob[merge_mask] = merge_prob[merge_mask]
    invalid_merge_mask = np.sum(merge_mask.reshape(merge_mask.shape[0], 49), axis = -1)==0
    new_merge_prob[invalid_merge_mask] = merge_prob[invalid_merge_mask]
    return new_merge_prob

def get_merge_depth_origin(ins_depth_pred, merge_prob):
    """按照论文的方法, 根据预测的深度和融合的深度的概率预测最终的深度

    Args:
        ins_depth_pred (_type_): 预测的深度
        merge_prob (_type_): 每个深度的概率
    Returns:
        _type_: 最终融合的深度
    """
    merge_depth = (np.sum((ins_depth_pred*merge_prob).reshape(-1, 49), axis=-1) /
                np.sum(merge_prob.reshape(-1, 49), axis=-1))
    return merge_depth

def draw_relation_between_vis_offset_uncer(pickle_info):
    """画出预测和真实数据的差与不确定性之间的关系

    Args:
        pickle_info (_type_): pickle文件中存储的数据
    """
    last_epoch_info = pickle_info[-1]
    total_vis_depth_offset = np.empty(0)
    total_vis_depth_uncer_pred = np.empty(0)
    
    for idx, batch_info in enumerate(last_epoch_info):
        vis_depth_pred = batch_info["vis_depth_pred"].reshape(-1)
        vis_depth_target = batch_info["vis_depth_target"].reshape(-1)
        vis_depth_uncer_pred = batch_info["vis_depth_uncertain_pred"].reshape(-1)
        
        total_vis_depth_offset = np.concatenate([total_vis_depth_offset, np.abs(vis_depth_pred-vis_depth_target)])
        total_vis_depth_uncer_pred = np.concatenate([total_vis_depth_uncer_pred, vis_depth_uncer_pred])
    plt.scatter(total_vis_depth_offset, total_vis_depth_uncer_pred, s=0.1)

def draw_last_epoch_offset(pickle_info):
    """画出预测的深度和真实深度的差

    Args:
        pickle_info (_type_): pickle文件中存储的数据
    """
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
        
        merge_prob = get_merge_prob_origin(ins_depth_uncer_pred)
        merge_depth = get_merge_depth_origin(ins_depth_pred, merge_prob).reshape(-1)
        
        depth_target = batch_info["depth_target"].reshape(-1)
        total_ins_depth_offset = np.concatenate([total_ins_depth_offset, merge_depth-depth_target])
        total_depth_target = np.concatenate([total_depth_target, depth_target])
        
    plt.scatter(total_depth_target, total_ins_depth_offset, s=0.1)

def savefig(fig_name):
    plt.savefig(fig_name)
    
    
if __name__ == "__main__":
    pickle_info = read_pickle_file("/home/public/zty/Project/DeepLearningProject/DID-M3D/record_transformer_200.pkl")
    draw_relation_between_vis_offset_uncer(pickle_info)
    xs = np.arange(0.1, 70, 0.1)
    draw_func(min_uncertainty_loss_log_variance, xs)
    savefig("relation_between_offset_uncer_transformer.png")
        
        
        

        
        