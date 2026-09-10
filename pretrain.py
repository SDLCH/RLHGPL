import os
import numpy
import copy
import torch
from pretrain.utils import load_data, set_params
from pretrain.module.meow import MEOW
from pretrain.module.preprocess import *
import random


args = set_params('acm')

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.device))
    torch.cuda.set_device(args.device)
else:
    device = torch.device("cpu")


seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def train():
    nei_index, feats, adjs, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]
    if args.dataset == 'aminer':
        feats_dim_list = [64,64,64]
    else:
        feats_dim_list = [i.shape[1] for i in feats]
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("The dim of different kinds' nodes' feature: ", feats_dim_list)
    feat = feats[0]
    adjs = pathsim(adjs, args.nei_max)
    mask_feat = mask_features(feat, args.feat_mask)
    adjs_norm = [normalize_adj(adj) for adj in adjs]
    mask_adjs = mask_edges(adjs, sub_num, args.adj_mask)
    print("Feature and Edge Mask Finished!")

    model = MEOW(feats_dim_list, sub_num, args.hidden_dim, args.embed_dim, args.tau, adjs_norm, args.lam_proto, \
                  args.dropout, nei_index, args.dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feat = feat.cuda()
        feats = [f.cuda() for f in feats]
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test]

    cnt_wait = 0
    best = 1e9
    best_t = 0
    best_model_state_dict = None


    epoch_times = args.nb_epochs

    num_clusters = args.num_cluster
    for epoch in range(epoch_times):
        model.train()
        optimizer.zero_grad()
        loss = model(feats, mask_feat, mask_adjs, adjs_norm, num_clusters)
        print(f'Epoch:{epoch}. total loss:{loss.item()}')
        loss.backward()
        optimizer.step()

        if best > loss.item():
            best = loss.item()
            best_t = epoch
            cnt_wait = 0
            best_model_state_dict = copy.deepcopy(model.state_dict())
        else:
            cnt_wait += 1

        if cnt_wait >= args.patience:
            print(f'Early stopping in {best_t}! loss:{best:.4f}')
            save_dir = "./pre_trained_models"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{args.dataset}.pth")
            torch.save(best_model_state_dict, save_path)
            print(f"+++model saved ! ./pre_trained_models/{args.dataset}.pth")
            break


if __name__ == '__main__':
    train()
