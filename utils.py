import torch
import numpy as np


def patch_loss_global(ytest, pids_test_patch, pids_train):
    criterion = torch.nn.CrossEntropyLoss()
    ytest =  ytest.view( ytest.size()[0],  ytest.size()[1], -1).transpose(1,2).reshape(-1, ytest.size()[1])
    pids_test_patch =  pids_test_patch.view(-1)
    pids_train_patch = pids_train.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11).view(-1)
    loss = criterion(ytest, torch.cat([pids_train_patch,pids_test_patch.view(-1)]))
    return loss

def patch_loss_local(labels_test_patch, cls_scores):
    criterion = torch.nn.CrossEntropyLoss()
    labels_test_patch = labels_test_patch.view(-1)
    cls_scores = cls_scores.view(cls_scores.size()[0], cls_scores.size()[1], -1).transpose(1,2).reshape(-1, cls_scores.size()[1])
    loss = criterion(cls_scores, labels_test_patch.view(-1))
    return loss

def generate_matrix():
    xd = np.random.randint(1, 2)
    yd = np.random.randint(1, 2)
    index = list(range(11))
    x0 = np.random.choice(index, size=xd, replace=False)
    y0 = np.random.choice(index, size=yd, replace=False)
    return x0, y0


def random_block(x):
    x0, y0 = generate_matrix()
    mask = torch.zeros([1, 1, 11, 11], requires_grad=False) +1
    for i in x0:
        for j in y0:
                mask[:, :, i, j] = 0
    mask = mask.float()
    x = x * mask.cuda()
    return x

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def patchmix(test_img, test_label, global_label):
    test_label_patch = test_label.unsqueeze(2).unsqueeze(2).repeat(1, 1,11, 11)
    global_label_patch = global_label.unsqueeze(2).unsqueeze(2).repeat(1, 1, 11, 11)

    batch_size = test_img.size()[0]
    for i in range(batch_size):
        test_label_patch_slice = test_label_patch[i]
        global_label_patch_slice = global_label_patch[i]
        input = test_img[i]
        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(input.size()[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        test_img[i] = input 

        #### calculate patch label
        bbx1, bby1, bbx2, bby2 = float(bbx1), float(bby1), float(bbx2), float(bby2)
        bbx1, bby1, bbx2, bby2 = round(bbx1* 11.0/84.0), round(bby1* 11.0/84.0), round(bbx2* 11.0/84.0), round(bby2 * 11.0/84.0)
        test_label_patch_slice[:, bbx1:bbx2, bby1:bby2] = test_label_patch_slice[rand_index, bbx1:bbx2, bby1:bby2]
        global_label_patch_slice[:, bbx1:bbx2, bby1:bby2] = global_label_patch_slice[rand_index, bbx1:bbx2, bby1:bby2]
        ### #############
        test_label_patch[i] = test_label_patch_slice 
        global_label_patch[i] = global_label_patch_slice 
     
    return test_img, test_label_patch, global_label_patch

#### from cross attention
def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


import os
import os.path as osp
import errno
import json
import shutil

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def save_checkpoint(state, is_best=False, fpath='checkpoint.pth.tar'):
    if len(osp.dirname(fpath)) != 0:
        mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot
