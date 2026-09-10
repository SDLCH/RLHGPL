import torch
import datetime
import numpy as np
import random
from pretrain.utils import load_data
from prompt.task import NodeTask
from prompt.params import *


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    np.set_printoptions(precision=2, floatmode='fixed')
    set_random_seed(args.seed)

    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    print("Dataset:", args.dataset)

    nei_index, feats, adjs, label, idx_train, idx_val, idx_test = load_data(args.dataset, args.ratio, args.type_num)
    if args.dataset == 'aminer':
        feats_dim_list = [64, 64, 64]
    else:
        feats_dim_list = [i.shape[1] for i in feats]
    sub_num = int(len(adjs))
    feat = feats[0]
    idx_val = idx_val[-1]
    idx_test = idx_test[-1]
    nei_index = [nei.to(device) for nei in nei_index]
    feats = [feat.to(device) for feat in feats]
    label = label.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    tasker = NodeTask(args=args, device=device, sub_num=sub_num, feats_dim_list=feats_dim_list, nei_index=nei_index,
                      adjs=adjs, feat=feat, feats=feats, label=label, idx_val=idx_val, idx_test=idx_test)
    return tasker


if __name__ == "__main__":
    args = get_args()
    print(args)
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
        torch.cuda.set_device(args.device)
    else:
        device = torch.device("cpu")

    task = main()
    task.run()

