import torch.optim as optim

def build_optimizer(cfg_optimizer, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [{'params': biases, 'weight_decay': 0},
                  {'params': weights, 'weight_decay': cfg_optimizer['weight_decay']}]
# def build_optimizer(cfg_optimizer, model):
#     weights, biases_not_in_norm, biases_in_norm, norm = [], [], [], []
#     for name, param in model.named_parameters():
#         if 'bias' in name and 'bn' in name:
#             biases_in_norm += [param]
#         elif 'bias' in name and 'bn' not in name:
#             biases_not_in_norm += [param]
#         elif 'bn' in name:
#             norm += [param]
#         else:
#             weights += [param]

#     parameters = [{'params': biases_in_norm, 'weight_decay': 0},
#                   {'params': biases_not_in_norm, 'weight_decay': 0, 'lr': cfg_optimizer['lr']*2},
#                   {'params': norm, 'weight_decay': 0},
#                   {'params': weights, 'weight_decay': cfg_optimizer['weight_decay']}]

    if cfg_optimizer['type'] == 'adam':
        optimizer = optim.Adam(parameters, lr=cfg_optimizer['lr'])
    elif cfg_optimizer['type'] == 'sgd':
        optimizer = optim.SGD(parameters, lr=cfg_optimizer['lr'], momentum=0.9)
    elif cfg_optimizer['type'] == 'adamw':
        optimizer = optim.AdamW(parameters, lr=cfg_optimizer['lr'], betas=(0.95, 0.99), weight_decay=0.00001)
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg_optimizer['type'])

    return optimizer